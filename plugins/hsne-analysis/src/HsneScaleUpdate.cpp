#include "HsneScaleUpdate.h"
#include "HsneHierarchy.h"

#include <algorithm>
#include <numeric>
#include <array>
#include <iterator>

/// ///////////////////// ///
/// HsneScaleUpdateWorker ///
/// ///////////////////// ///

HsneScaleUpdateWorker::HsneScaleUpdateWorker(const HsneHierarchy& hsneHierarchy) :
    _hsneHierarchy(hsneHierarchy),
    _embedding(nullptr),
    _roi(nullptr),
    _imageIndices(nullptr),
    _mappingBottomToLocal(nullptr),
    _mappingLocalToBottom(nullptr),
    _idMap(),
    _tresh_influence(-1.0f),
    _currentScaleLevel(0),
    _newScaleLevel(0),
    _fixScale(false),
    _traversalDirection(utils::TraversalDirection::AUTO),
    _landmarkFilterNumber(0),
    _embScalingFactors(),
    _IdRoiRepresentation(),
    _initEmbedding(nullptr),
    _initTypes(),
    _newTransitionMatrix(nullptr)
{

}

void HsneScaleUpdateWorker::setData(Dataset<Points> embedding, const utils::ROI& roi, const Eigen::MatrixXui& imageIndices, IDMapping& idMap, const bool fixScale, 
    const float tresh_influence, const utils::VisualBudgetRange visualBudget, const std::pair<float, float> embScalingFactors, const utils::EmbeddingExtends currentEmbExtends, 
    uint32_t landmarkFilterNumber, const utils::TraversalDirection direction, LandmarkMapSingle& mappingBottomToLocal, LandmarkMap& mappingLocalToBottom,
    std::vector<float>& initEmbedding, HsneMatrix& transitionMatrix)
{
    _embedding = embedding;
    _roi = &roi;
    _imageIndices = &imageIndices;
    _idMap = &idMap;
    _fixScale = fixScale;
    _tresh_influence = tresh_influence;
    _landmarkFilterNumber = landmarkFilterNumber;
    _visualBudget = visualBudget;
    _embScalingFactors = embScalingFactors;
    _currentEmbExtends = currentEmbExtends;
    _mappingBottomToLocal = &mappingBottomToLocal;
    _mappingLocalToBottom = &mappingLocalToBottom;
    _traversalDirection = direction;
    _initEmbedding = &initEmbedding;
    _newTransitionMatrix = &transitionMatrix;
}


void HsneScaleUpdateWorker::updateScale()
{
    Log::info("HsneScaleUpdateWorker::updateScale()");
    utils::ScopedTimer updateScaleTimer("Total scale update");

    // Get selecion IDs in current viewport on the image
    std::vector<uint32_t> imageSelectionIDs;
    utils::timer([&]() {
        utils::extractIdBlock(_roi->layerBottomLeft, _roi->layerTopRight, *_imageIndices, imageSelectionIDs);
        },
        "selecion IDs in current viewport");

    if (_traversalDirection == utils::TraversalDirection::AUTO)
    {
        // Local indices on scale: Go up from bottom (image ID selection) to refinedScaleLevel or stay on fixed scale (if set in UI)
        utils::timer([&]() {
            if (_fixScale)
            {
                _newScaleLevel = _currentScaleLevel;

                if (_tresh_influence == -1.0f)
                    utils::computeLocalIDsOnCoarserScaleHeuristic(_newScaleLevel, imageSelectionIDs, _hsneHierarchy, _localIDsOnNewScale);
                else
                    utils::computeLocalIDsOnCoarserScale(_currentScaleLevel, imageSelectionIDs, _hsneHierarchy, _tresh_influence, _localIDsOnNewScale);
            }
            else
            {
                _newScaleLevel = 0;
                utils::localIDsOnCoarserScale(utils::VisualTarget(_visualBudget), imageSelectionIDs, _hsneHierarchy, _tresh_influence, _newScaleLevel, _localIDsOnNewScale);
            }
            },
            "computeLocalIDs");
    }
    else
    {
        utils::timer([&]() {
            _newScaleLevel = _currentScaleLevel;
            utils::applyTraversalDirection(_traversalDirection, _newScaleLevel);
            utils::computeLocalIDsOnCoarserScaleHeuristic(_newScaleLevel, imageSelectionIDs, _hsneHierarchy, _localIDsOnNewScale);
            },
            "computeLocalIDsOnCoarserScaleHeuristic");
    }

    emit scaleLevelComputed(_newScaleLevel);

    Log::info("HsneScaleUpdateWorker::updateScale: " + std::to_string(_localIDsOnNewScale.size()) + " landmarks on scale " + std::to_string(_newScaleLevel) +
        " (previously scale " + std::to_string(_currentScaleLevel) + ") for " + std::to_string(imageSelectionIDs.size()) + " data points in view");


    // Compute the transition matrix for the landmarks above the threshold
    utils::timer([&]() {
            _hsneHierarchy.getTransitionMatrixForSelectionAtScale(_newScaleLevel, _landmarkFilterNumber, _localIDsOnNewScale, *_newTransitionMatrix);
        },
        "getTransitionMatrixForSelectionAtScale");

    // Compute landmarkRoiRepresentation: To what extend do the landmarks represent data points that are in roi vs outside
    utils::timer([&]() {
        utils::landmarkRoiRepresentation(_imgSize, *_roi, _hsneHierarchy, _newScaleLevel, _localIDsOnNewScale, _IdRoiRepresentation);
        },
        "landmarkRoiRepresentation");

    // Rescale embedding every update
    std::vector<mv::Vector2f> embPosRescaled;
    utils::EmbeddingExtends embExtendsRescaled;
    utils::timer([&]() {
        utils::rescaleEmbedding(_embedding, _embScalingFactors, _currentEmbExtends, embPosRescaled, embExtendsRescaled);
        },
        "rescaleEmbedding");

    // Use previous embedding as init of new embedding
    utils::timer([&]() {
        utils::reinitializeEmbedding(_hsneHierarchy, embPosRescaled, *_idMap, embExtendsRescaled, _newScaleLevel, _localIDsOnNewScale, *_initEmbedding, _initTypes);
        },
        "reinitializeEmbedding");

    // new ID mapping 
    utils::timer([&]() {
        utils::recomputeIDMap(_hsneHierarchy.getScale(_newScaleLevel), _localIDsOnNewScale, *_idMap);
        },
        "new ID mapping");

    // selection map at scale based on ID mapping
    utils::timer([&]() {
        _hsneHierarchy.computeSelectionMapsAtScale(_newScaleLevel, _localIDsOnNewScale, *_mappingBottomToLocal, *_mappingLocalToBottom);
        },
        "selection map at scale based on ID mapping");

    Log::info("#selected image indices: " + std::to_string(imageSelectionIDs.size()));
    Log::info("#corresponding landmarks at current scale: " + std::to_string(_localIDsOnNewScale.size()));
    Log::info("Refining embedding...");

    _currentScaleLevel = _newScaleLevel;

    emit finished(true);
}

HsneScaleUpdateWorker::~HsneScaleUpdateWorker() {
    Log::debug("~HsneScaleWorker");
}

std::vector<float> HsneScaleUpdateWorker::getRoiRepresentationFractions() const
{
    std::vector<float> roiRepresentationFractions;
    roiRepresentationFractions.reserve(_IdRoiRepresentation.size());

    for (size_t n = 0; n < _IdRoiRepresentation.size(); ++n)
        roiRepresentationFractions.emplace_back(_IdRoiRepresentation[n].first);

    return roiRepresentationFractions;
}

std::vector<float> HsneScaleUpdateWorker::getNumberTransitions() const
{
    std::vector<float> numberTransitions;
    numberTransitions.reserve(_newTransitionMatrix->size());

    for (size_t n = 0; n < _newTransitionMatrix->size(); ++n)
        numberTransitions.emplace_back((*_newTransitionMatrix)[n].size());

    return numberTransitions;
}

std::vector<float> HsneScaleUpdateWorker::getInitTypesAsFloats() const
{
    std::vector<float> initTypes;
    initTypes.reserve(_initTypes.size());

    for (size_t n = 0; n < _initTypes.size(); ++n)
        initTypes.emplace_back(static_cast<float>(_initTypes[n]));

    return initTypes;
}


/// /////////////// ///
/// HsneScaleUpdate ///
/// /////////////// ///

HsneScaleUpdate::HsneScaleUpdate(const HsneHierarchy& hsneHierarchy) :
    _workerThread(),
    _hsneScaleWorker(nullptr),
    _isRunning(false)
{

    _hsneScaleWorker = new HsneScaleUpdateWorker(hsneHierarchy);
    _hsneScaleWorker->moveToThread(&_workerThread);

    // To-Worker signals
    connect(this, &HsneScaleUpdate::startWorker, _hsneScaleWorker, &HsneScaleUpdateWorker::updateScale);

    // From-Worker signals
    connect(_hsneScaleWorker, &HsneScaleUpdateWorker::started, this, [this]() { _isRunning = true; });
    connect(_hsneScaleWorker, &HsneScaleUpdateWorker::finished, this, [this](bool success) { 
        _isRunning = false;
        emit finished(success);
    });

    // re-emit the signal for the outside world
    connect(_hsneScaleWorker, &HsneScaleUpdateWorker::scaleLevelComputed, this, &HsneScaleUpdate::scaleLevelComputed);

    _workerThread.start();

}

HsneScaleUpdate::~HsneScaleUpdate()
{
}

void HsneScaleUpdate::startComputation(Dataset<Points> embedding, const utils::ROI& roi, const Eigen::MatrixXui& imageIndices, IDMapping& idMap, bool fixScale,
    const float tresh_influence, const utils::VisualBudgetRange visualBudgetRange, const std::pair<float, float> embScalingFactors, const utils::EmbeddingExtends currentEmbExtends, 
    uint32_t landmarkFilterNumber, const utils::TraversalDirection direction, LandmarkMapSingle& mappingBottomToLocal, LandmarkMap& mappingLocalToBottom,
    std::vector<float>& initEmbedding, HsneMatrix& transitionMatrix)
{
    _hsneScaleWorker->setData(embedding, roi, imageIndices, idMap, fixScale, tresh_influence, visualBudgetRange, embScalingFactors, currentEmbExtends, 
        landmarkFilterNumber, direction, mappingBottomToLocal, mappingLocalToBottom, initEmbedding, transitionMatrix);
    emit startWorker();
}

