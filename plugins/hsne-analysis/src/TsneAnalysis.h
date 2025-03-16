#pragma once

#include "TsneParameters.h"
#include "TsneData.h"
#include "Utils.h"
Q_DECLARE_METATYPE(utils::EmbeddingExtends);

#include "hdi/dimensionality_reduction/hd_joint_probability_generator.h"
#pragma warning( push ) 
#pragma warning( disable : 4267 ) // disable 'size_t' to 'uint32_t' warning from external library
#include "hdi/dimensionality_reduction/gradient_descent_tsne_texture.h"
#pragma warning( pop ) 

#include <QThread>
#include <QPointer>

#include <vector>
#include <string>
#include <map>

class OffscreenBuffer;
class TsneAnalysis;

class TsneWorker : public QObject
{
    Q_OBJECT
public:
    TsneWorker(const TsneParameters& parameters, OffscreenBuffer* buffer, TsneData* _outEmd, const HsneMatrix& probDist, std::vector<float>& inital_embedding, uint32_t numPoints);
    TsneWorker(const TsneParameters& parameters, OffscreenBuffer* buffer, TsneData* _outEmd, const HsneMatrix& probDist, uint32_t numPoints);
    TsneWorker(const TsneParameters& parameters, OffscreenBuffer* buffer, TsneData* _outEmd, /*const*/ std::vector<float>& data, uint32_t numDimensionsData);
    ~TsneWorker();

    uint32_t getNumIterations() const {return _currentIteration + 1u; }
    const size_t getWorkerID() const { return _workerID; }

    void setName(const  std::string& name) { _analysisParentName = name; }
    std::string getName() const { return _analysisParentName; }

public slots:
    void compute();
    void continueComputation(uint32_t iterations);
    void stop();

signals:
    void embeddingUpdate(const std::vector<float>& emb, const uint32_t& numPoints, const uint32_t& numDimensions);
    void finished();
    void publishExtends(utils::EmbeddingExtends extends);

private:
    void computeSimilarities();
    void computeGradientDescent(uint32_t iterations);
    
private:
    /** Parameters for the execution of the similarity computation and gradient descent */
    // TODO: use a single parameters instance instead of copying all the parameters
    TsneParameters _parameters;

    /** Current iteration in the embedding / gradient descent process */
    uint32_t _currentIteration;

    // Data variables
    uint32_t  _numPoints;
    uint32_t  _numDimensionsData;

    /** High-dimensional input data */
    std::vector<float>* _data;

    /** High-dimensional probability distribution encoding point similarities */
    const HsneMatrix* _probabilityDistributionGiven;
    HsneMatrix _probabilityDistributionLocal;

    /** Check if the worker was initialized with a probability distribution or data */
    bool _hasProbabilityDistribution;

    /** GPGPU t-SNE gradient descent implementation */
    hdi::dr::GradientDescentTSNETexture<HsneMatrix> _GPGPU_tSNE;

    /** Storage of current embedding */
    hdi::data::Embedding<float> _embedding;

    /** Transfer embedding data array */
    TsneData* _outEmbedding;

    /** Offscreen OpenGL buffer required to run the gradient descent */
    OffscreenBuffer* _offscreenBuffer;

    // Termination flags
    bool _shouldStop;

    // Debugging counter
    size_t _workerID;
    static size_t _workerCount;

    // Name for logging
    std::string _analysisParentName;
};

class TsneAnalysis : public QObject
{
    Q_OBJECT
public:
    TsneAnalysis(std::string name = "");
    ~TsneAnalysis() override;

    void startComputation(const TsneParameters& parameters, const HsneMatrix& probDist, std::vector<float>& inital_embedding, uint32_t numPoints);
    void startComputation(const TsneParameters& parameters, const HsneMatrix& probDist, uint32_t numPoints);
    void startComputation(const TsneParameters& parameters, /*const*/ std::vector<float>& data, uint32_t numDimensionsData);

    void continueComputation(uint32_t iterations);
    void stopComputation();

    bool canContinue() const { return (_tsneWorker == nullptr) ? false : _tsneWorker->getNumIterations() >= 1; }
    uint32_t getNumIterations() const { return _tsneWorker->getNumIterations(); }
    const TsneData& getEmbedding() const { return _embedding; }
    bool threadIsRunning() const { return _workerThread.isRunning(); }

private:
    void startComputation(TsneWorker* tsneWorker);

signals:
    // Local signals
    void startWorker();
    void continueWorker(uint32_t iterations);
    void stopWorker();

    // Outgoing signals
    void embeddingUpdate(const std::vector<float>& emb, const uint32_t& numPoints, const uint32_t& numDimensions);
    void finished();
    void publishExtends(utils::EmbeddingExtends extends);

private:
    QThread                 _workerThread;
    std::string             _analysisName;
    QPointer<TsneWorker>    _tsneWorker;

    TsneData                _embedding;

    /** Offscreen OpenGL buffer required to run the gradient descent */
    QPointer<OffscreenBuffer> _offscreenBuffer;
};
