#pragma once

#include "PointData/PointData.h"
#include "CommonTypes.h"
#include "Utils.h"
#include "UtilsScale.h"

#include <QThread>
#include <QPointer>

using namespace mv;

class HsneHierarchy;

/**
 * HSNE interactive scale worker class
 *
 * TODO: Refactor -> set pointers already in constructor and not in setData()
 * 
 * @author Alexander Vieth
 */
class HsneScaleUpdateWorker : public QObject
{
    Q_OBJECT
public:
    HsneScaleUpdateWorker(const HsneHierarchy& hsneHierarchy);
    ~HsneScaleUpdateWorker();

    // Setter

    void setData(Dataset<Points> embedding, const utils::ROI& roi, const Eigen::MatrixXui& imageIndices, IDMapping& idMap, const bool fixScale,
        const float tresh_influence, const utils::VisualBudgetRange visualBudget, const std::pair<float, float> embScalingFactors, const utils::EmbeddingExtends currentEmbExtends,
        uint32_t landmarkFilterNumber, const utils::TraversalDirection direction, LandmarkMapSingle& mappingBottomToLocal, LandmarkMap& mappingLocalToBottom,
        std::vector<float>& initEmbedding, HsneMatrix& transitionMatrix);

    void setImageSize(QSize imgSize) {
        _imgSize = imgSize;
    }

    void setInitalTopLevelScale(uint32_t scale) {
        _currentScaleLevel = scale;
        _newScaleLevel = scale;
    }

    // Getter
    std::vector<uint32_t> getLocalIDsOnNewScale() const { return _localIDsOnNewScale; }
    std::vector<float> getRoiRepresentationFractions() const;
    std::vector<float> getNumberTransitions() const;
    std::vector<float> getInitTypesAsFloats() const;
    uint32_t getCurrentScaleLevel() const { return _currentScaleLevel; }

public slots:
    /** Update the landmarks in the embedding based on the current viewport selection in the image */
    void updateScale();

signals:
    void started();
    void finished(bool success);
    void scaleLevelComputed(uint32_t);

private:
    Dataset<Points>                 _embedding;
    QSize                           _imgSize;

    const HsneHierarchy&            _hsneHierarchy;
    const utils::ROI*               _roi;
    const Eigen::MatrixXui*         _imageIndices;
    float                           _tresh_influence;       // could be used in computeLocalIDsOnCoarserScale

    std::vector<uint32_t>           _localIDsOnNewScale;
    HsneMatrix*                     _newTransitionMatrix;

    IDMapping*                      _idMap;                 /** Maps data IDs (that are currently represented) to embdding points' localIdOnScale & posInEmbedding  */
    LandmarkMapSingle*              _mappingBottomToLocal;
    LandmarkMap*                    _mappingLocalToBottom;

    uint32_t                        _currentScaleLevel;     /** The scale the current embedding is a part of */
    uint32_t                        _newScaleLevel;         /** The scale the next embedding is a part of */
    bool                            _fixScale;              /** Don't traverse the scale, reuse previous level */
    utils::TraversalDirection       _traversalDirection;

    uint32_t                        _landmarkFilterNumber;  /** Number of transitions a landmark must have to remain, 0 means no filtering */

    utils::EmbeddingExtends         _currentEmbExtends;
    std::pair<float, float>         _embScalingFactors;
    utils::VisualBudgetRange        _visualBudget;

    std::vector<std::pair<float, std::vector<uint32_t>>> _IdRoiRepresentation;

    std::vector<float>*             _initEmbedding;
    std::vector<utils::POINTINITTYPE>_initTypes;           /** init type of embedding points */
};


/**
 * HSNE interactive scale update class
 * *
 * @author Alexander Vieth
 */
class HsneScaleUpdate : public QObject
{
    Q_OBJECT
public:
    HsneScaleUpdate(const HsneHierarchy& hsneHierarchy);
    ~HsneScaleUpdate();

    void startComputation(Dataset<Points> embedding, const utils::ROI& roi, const Eigen::MatrixXui& imageIndices, IDMapping& idMap, const bool fixScale,
        const float tresh_influence, const utils::VisualBudgetRange visualBudget, const std::pair<float, float> embScalingFactors, const utils::EmbeddingExtends currentEmbExtends,
        uint32_t landmarkFilterNumber, const utils::TraversalDirection direction, LandmarkMapSingle& mappingBottomToLocal, LandmarkMap& mappingLocalToBottom,
        std::vector<float>& initEmbedding, HsneMatrix& transitionMatrix);

    // Setter
    void setImageSize(QSize imgSize) { _hsneScaleWorker->setImageSize(imgSize); }
    void setInitalTopLevelScale(uint32_t scale) { _hsneScaleWorker->setInitalTopLevelScale(scale); }

    // Getter
    std::vector<uint32_t> getLocalIDsOnNewScale() const { return _hsneScaleWorker->getLocalIDsOnNewScale();}
    std::vector<float> getInitTypes() const { return _hsneScaleWorker->getInitTypesAsFloats(); }    // transforms utils::POINTINITTYPE to float
    std::vector<float> getRoiRepresentationFractions() const { return _hsneScaleWorker->getRoiRepresentationFractions(); }
    std::vector<float> getNumberTransitions() const { return _hsneScaleWorker->getNumberTransitions(); }
    
    bool isRunning() const { return _isRunning; }

signals:
    // Local signals
    void startWorker();
    void stopWorker();

    // Outgoing signals
    void finished(uint32_t success);
    void scaleLevelComputed(uint32_t);

private:
    QThread                         _workerThread;
    QPointer<HsneScaleUpdateWorker> _hsneScaleWorker;
    bool                            _isRunning;

};
