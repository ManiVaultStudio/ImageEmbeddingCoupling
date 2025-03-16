#include "TsneAnalysis.h"

#include "hdi/utils/glad/glad.h"
#include "OffscreenBuffer.h"

#include <vector>
#include <assert.h>

#include "Logger.h"

#include "hdi/dimensionality_reduction/tsne_parameters.h"

/// /////////// ///
/// TSNE WORKER ///
/// /////////// ///

size_t TsneWorker::_workerCount = 0;

TsneWorker::TsneWorker(const TsneParameters& parameters, OffscreenBuffer* buffer, TsneData* _outEmd, const HsneMatrix& probDist, std::vector<float>& inital_embedding, uint32_t numPoints):
    _currentIteration(0),
    _parameters(parameters),
    _data(nullptr),
    _probabilityDistributionGiven(&probDist),
    _probabilityDistributionLocal(),
    _numPoints(numPoints),
    _numDimensionsData(0),
    _hasProbabilityDistribution(true),
    _shouldStop(false),
    _workerID(++_workerCount),
    _analysisParentName(""),
    _offscreenBuffer(buffer),
    _outEmbedding(_outEmd)
{
    // Use inital embedding
    _embedding.resize(2, numPoints);
    _embedding.getContainer().assign(inital_embedding.begin(), inital_embedding.end());
    _parameters.setHasPresetEmbedding(true);

    if (_probabilityDistributionGiven == nullptr)
        Log::critical("TsneWorker::TsneWorker: _probabilityDistributionGiven is nullptr");
}

TsneWorker::TsneWorker(const TsneParameters& parameters, OffscreenBuffer* buffer, TsneData* _outEmd, const HsneMatrix& probDist, uint32_t numPoints) :
    _currentIteration(0),
    _parameters(parameters),
    _data(nullptr),
    _probabilityDistributionGiven(&probDist),
    _probabilityDistributionLocal(),
    _numPoints(numPoints),
    _numDimensionsData(0),
    _hasProbabilityDistribution(true),
    _shouldStop(false),
    _workerID(++_workerCount),
    _analysisParentName(""),
    _offscreenBuffer(buffer),
    _outEmbedding(_outEmd)
{
    if (_probabilityDistributionGiven == nullptr)
        Log::critical("TsneWorker::TsneWorker: _probabilityDistributionGiven is nullptr");
}

TsneWorker::TsneWorker(const TsneParameters& parameters, OffscreenBuffer* buffer, TsneData* _outEmd, /*const*/ std::vector<float>& data, uint32_t numDimensionsData) :
    _currentIteration(0),
    _parameters(parameters),
    _data(&data),
    _probabilityDistributionGiven(nullptr),
    _probabilityDistributionLocal(),
    _numPoints(static_cast<uint32_t>(data.size()) / numDimensionsData),
    _numDimensionsData(numDimensionsData),
    _hasProbabilityDistribution(false),
    _shouldStop(false),
    _workerID(++_workerCount),
    _analysisParentName(""),
    _offscreenBuffer(buffer),
    _outEmbedding(_outEmd)
{
}

TsneWorker::~TsneWorker()
{
    Log::info("TsneWorker::~TsneWorker " + std::to_string(_workerID) + " (" + _analysisParentName + ")");
}

void TsneWorker::computeSimilarities()
{
    if (_numDimensionsData == 0)
        Log::critical("TsneWorker::computeSimilarities: Number of data dimension is 0. Cannot computer high-dimensional similarities");

    hdi::dr::HDJointProbabilityGenerator<float>::Parameters probGenParams;
    probGenParams._perplexity = _parameters.getPerplexity();
    probGenParams._perplexity_multiplier = 3;
    probGenParams._aknn_annoy_num_trees = _parameters.getNumTrees();
    probGenParams._aknn_hnsw_eff = _parameters.getHNSWeff();
    probGenParams._aknn_hnsw_M = _parameters.getHNSWM();
    probGenParams._aknn_algorithm = _parameters.getKnnAlgorithm();
    probGenParams._aknn_metric = _parameters.getKnnDistanceMetric();

    Log::info("TsneWorker::computeSimilarities: tSNE initialized.");

    Log::info(fmt::format("TsneWorker::computeSimilarities: Computing high dimensional probability distributions. Num dims: {0}, Num data points: {1}" , _numDimensionsData, _numPoints));
    Log::debug(fmt::format("TsneWorker::computeSimilarities: Use knn algorithm {}", static_cast<int>(probGenParams._aknn_algorithm)));
    hdi::dr::HDJointProbabilityGenerator<float> probabilityGenerator;
    {
        utils::ScopedTimer gradDescentTimer("A-tSNE probability distribution");
        _probabilityDistributionLocal.clear();
        _probabilityDistributionLocal.resize(_numPoints);
        probabilityGenerator.computeProbabilityDistributions(_data->data(), _numDimensionsData, _numPoints, _probabilityDistributionLocal, probGenParams);
    }

    Log::info("TsneWorker::computeSimilarities: Probability distributions calculated.");
}

void TsneWorker::computeGradientDescent(uint32_t iterations)
{
    if (_shouldStop)
        return;

    if (iterations <= 0)
    {
        Log::error("TsneWorker::computeGradientDescent: Number of iterations must be >0");
        return;
    }

    if (iterations < _currentIteration)
    {
        Log::error("TsneWorker::computeGradientDescent: Must continue with iterations > currentIterations");
        return;
    }

    hdi::dr::TsneParameters tsneParameters;

    tsneParameters._embedding_dimensionality = _parameters.getNumDimensionsOutput();
    tsneParameters._mom_switching_iter = _parameters.getExaggerationIter();
    tsneParameters._remove_exaggeration_iter = _parameters.getExaggerationIter();
    tsneParameters._exponential_decay_iter = _parameters.getExponentialDecayIter();
    tsneParameters._exaggeration_factor = (_parameters.getExaggerationFactor() != -1) ? _parameters.getExaggerationFactor() : 4 + _numPoints / 60000.0;
    tsneParameters._presetEmbedding = _parameters.getHasPresetEmbedding();

    Log::info(fmt::format("TsneWorker::computeGradientDescent: t-SNE settings: Exaggeration factor {0}, exaggeration iterations {1}, exponential decay iter {2}", 
        tsneParameters._exaggeration_factor, tsneParameters._remove_exaggeration_iter, tsneParameters._exponential_decay_iter));

    // Initialize GPU gradient descent
    {
        Log::info("TsneWorker::computeGradientDescent: Initialize GPU gradient descent.");
        utils::ScopedTimer gradDescentTimer("Initialize GPU gradient descent");

        // Create a context local to this thread that shares with the global share context
        if (!_offscreenBuffer->isInitialized())
            _offscreenBuffer->initialize();
        _offscreenBuffer->bindContext();

        if (_currentIteration == 0)
        {
            if(_hasProbabilityDistribution)
                _GPGPU_tSNE.initialize(*_probabilityDistributionGiven, &_embedding, tsneParameters);
            else
                _GPGPU_tSNE.initialize(_probabilityDistributionLocal, &_embedding, tsneParameters);
        }

        emit embeddingUpdate(_embedding.getContainer(), _numPoints, _parameters.getNumDimensionsOutput());
    }
    
    // Computing gradient descent on GPU
    {
        Log::info("TsneWorker::computeGradientDescent: Computing gradient descent on GPU.");
        utils::ScopedTimer gradDescentTimer("Computing gradient descent on GPU");

        const auto beginIteration   = _currentIteration;
        const auto endIteration     = iterations;

        // Performs gradient descent for every iteration
        for (_currentIteration = beginIteration; _currentIteration < endIteration; ++_currentIteration)
        {
            // Perform a GPGPU-SNE iteration
            _GPGPU_tSNE.doAnIteration();

            if (_currentIteration > 0 && _currentIteration % 10 == 0)
                emit embeddingUpdate(_embedding.getContainer(), _numPoints, _parameters.getNumDimensionsOutput());

            if ((_currentIteration == _parameters.getPublishExtendsAtIteration()) && (_parameters.getPublishExtendsAtIteration() > 0))
            {
                Log::info("TsneWorker::computeGradientDescent: Set reference embedding extends at iteration " + std::to_string(_currentIteration));
                emit publishExtends(utils::computeExtends(_embedding.getContainer()));
            }

            // React to requests to stop
            if (_shouldStop)
                break;
        }

        _offscreenBuffer->releaseContext();

        emit embeddingUpdate(_embedding.getContainer(), _numPoints, _parameters.getNumDimensionsOutput());
    }

    _outEmbedding->assign(_numPoints, _parameters.getNumDimensionsOutput(), _embedding.getContainer());

    Log::info(fmt::format("TsneWorker::computeGradientDescent: Finished embedding of tSNE Analysis after: {} iterations", _currentIteration));
    emit finished();
}

void TsneWorker::compute()
{
    Log::info(fmt::format("A-tSNE: compute worker {0} ({1})", _workerID, _analysisParentName));
    utils::ScopedTimer computeTSNETimer("Total t-SNE computation");

    _shouldStop = false;

    if (!_hasProbabilityDistribution)
        computeSimilarities();

    computeGradientDescent(_parameters.getNumIterations());
}

void TsneWorker::continueComputation(uint32_t iterations)
{
    _shouldStop = false;

    computeGradientDescent(iterations);
}

void TsneWorker::stop()
{
    _shouldStop = true;
}

/// ///////////// ///
/// TSNE ANALYSIS ///
/// ///////////// ///

TsneAnalysis::TsneAnalysis(std::string name) :
    _workerThread(),
    _tsneWorker(nullptr),
    _analysisName(name),
    _offscreenBuffer(nullptr),
    _embedding()
{
    qRegisterMetaType<TsneData>();
    qRegisterMetaType<utils::EmbeddingExtends>();

    // Offscreen buffer must be created in the UI thread because it is a QWindow, afterwards we move it
    _offscreenBuffer = new OffscreenBuffer();

    // Move the Offscreen buffer to the processing thread after creating it in the UI Thread
    _offscreenBuffer->moveToThread(&_workerThread);
    _offscreenBuffer->getContext()->moveToThread(&_workerThread);
}

TsneAnalysis::~TsneAnalysis()
{
}

void TsneAnalysis::startComputation(const TsneParameters& parameters, const HsneMatrix& probDist, std::vector<float>& inital_embedding, uint32_t numPoints)
{
    if (!_tsneWorker.isNull())
    {
        _tsneWorker->deleteLater();
        _tsneWorker.clear();
    }

    _tsneWorker = new TsneWorker(parameters, _offscreenBuffer, &_embedding, probDist, inital_embedding, numPoints);
    startComputation(_tsneWorker);
}

void TsneAnalysis::startComputation(const TsneParameters& parameters, const HsneMatrix& probDist, uint32_t numPoints)
{
    if (_tsneWorker)
    {
        _tsneWorker->deleteLater();
        _tsneWorker.clear();
    }

    _tsneWorker = new TsneWorker(parameters, _offscreenBuffer, &_embedding, probDist, numPoints);
    startComputation(_tsneWorker);
}

void TsneAnalysis::startComputation(const TsneParameters& parameters, /*const*/ std::vector<float>& data, uint32_t numDimensionsData)
{
    if (_tsneWorker)
    {
        _tsneWorker->deleteLater();
        _tsneWorker.clear();
    }

    _tsneWorker = new TsneWorker(parameters, _offscreenBuffer, &_embedding, data, numDimensionsData);
    startComputation(_tsneWorker);
}

void TsneAnalysis::continueComputation(uint32_t iterations)
{
    emit continueWorker(iterations);
}

void TsneAnalysis::stopComputation()
{
    if (_workerThread.isRunning())
        Log::info("TsneAnalysis::stopComputation: about to stop tSNE computation of worker " + std::to_string(_tsneWorker->getWorkerID()));

    emit stopWorker();
    _workerThread.quit();
}

void TsneAnalysis::startComputation(TsneWorker* tsneWorker)
{
    tsneWorker->setName(_analysisName);
    tsneWorker->moveToThread(&_workerThread);

    // To-Worker signals
    connect(this, &TsneAnalysis::startWorker, tsneWorker, &TsneWorker::compute);
    connect(this, &TsneAnalysis::continueWorker, tsneWorker, &TsneWorker::continueComputation);
    connect(this, &TsneAnalysis::stopWorker, tsneWorker, &TsneWorker::stop, Qt::DirectConnection);

    // From-Worker signals
    connect(tsneWorker, &TsneWorker::embeddingUpdate, this, &TsneAnalysis::embeddingUpdate);
    connect(tsneWorker, &TsneWorker::finished, this, &TsneAnalysis::finished);
    connect(tsneWorker, &TsneWorker::publishExtends, this, &TsneAnalysis::publishExtends);

    _workerThread.start();

    emit startWorker();
}
