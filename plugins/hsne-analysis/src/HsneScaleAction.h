#pragma once

#include "actions/ToggleAction.h"
#include "actions/DecimalAction.h"
#include "actions/IntegralAction.h"
#include "actions/StatusAction.h"
#include "actions/TriggerAction.h"
#include "actions/ColorMap2DAction.h"

#include "TsneAnalysis.h"
#include "HsneScaleUpdate.h"
#include "PointData/PointData.h"
#include "CommonTypes.h"
#include "Utils.h"
#include "RecolorAction.h"

using namespace mv;
using namespace mv::gui;
using namespace mv::util;

class QMenu;
class QSize;

class InteractiveHsnePlugin;
class HsneHierarchy;
class TsneSettingsAction;

enum class NoUpdate {
    ISRUNNING,
    ROINOTGOODFORUPDATE,
    SEITINUI
};

/// ////////////////// ///
/// GUI: Scale Up&Down ///
/// ////////////////// ///

class ScaleDownUpActions : public WidgetAction
{
protected:

    /** Widget class for TSNE computation action */
    class Widget : public WidgetActionWidget {
    public:

        /**
         * Constructor
         * @param parent Pointer to parent widget
         * @param tsneComputationAction Pointer to TSNE computation action
         */
        Widget(QWidget* parent, ScaleDownUpActions* scaleDownUpActions);
    };

    /**
     * Get widget representation of the TSNE computation action
     * @param parent Pointer to parent widget
     * @param widgetFlags Widget flags for the configuration of the widget (type)
     */
    QWidget* getWidget(QWidget* parent, const std::int32_t& widgetFlags) override {
        return new Widget(parent, this);
    };

public:

    /**
     * Constructor
     * @param parent Pointer to parent object
     */
    ScaleDownUpActions(QObject* parent);

    void setNumScales(size_t numScales) { _numScales = numScales; }

    void currentScaleChanged(size_t currentScale);

public: // Action getters

    TriggerAction& getScaleUpAction() { return _scaleUpAction; }
    TriggerAction& getScaleDownAction() { return _scaleDownAction; }

protected:
    TriggerAction           _scaleUpAction;         /** Go a scale up action */
    TriggerAction           _scaleDownAction;       /** Go a scale down action */

    size_t                  _numScales;             /** Total number scales*/
};


/// /////////////// ///
/// HsneScaleAction ///
/// /////////////// ///

/**
 * HSNE interactive scale action class
 *
 * Action class for HSNE scale
 *
 * @author Thomas Kroes, Alexander Vieth
 */
class HsneScaleAction : public GroupAction
{
    Q_OBJECT
public:

    /**
     * Constructor
     * @param parent Pointer to parent object
     * @param tsneSettingsAction Reference to TSNE settings action
     * @param hsneHierarchy Reference to HSNE hierarchy
     * @param inputDataset Smart pointer to input dataset
     * @param embeddingDataset Smart pointer to embedding dataset
     * @param selectionSubset Smart pointer to selection subset of input data as created in InteractiveHsnePlugin::init()
     */
    HsneScaleAction(QObject* parent, InteractiveHsnePlugin* hsneAnalysisPlugin, TsneSettingsAction& tsneSettingsAction, HsneHierarchy& hsneHierarchy, 
                    Dataset<Points> inputDataset, Dataset<Points> embeddingDataset, Dataset<Points> firstEmbedding, Dataset<Points> topLevelLandmarkData,
                    Dataset<Points> pointInitTypesDataset, Dataset<Points> roiRepresentationDataset, Dataset<Points> numberTransitions, Dataset<Points> colorScatterRoiHSNE,
                    Dataset<Points> topLevelEmbedding);
    ~HsneScaleAction();

    /**
     * Get the context menu for the action
     * @param parent Parent widget
     * @return Context menu
     */
    QMenu* getContextMenu(QWidget* parent = nullptr) override;

    void computeTopLevelEmbedding();

    void update();

    void recomputeScaleEmbedding(bool randomInitMeta=false);
    
    /* Without changing the viewport, go up of down the hierarchy */
    void traverseHierarchyForView(utils::TraversalDirection direction);

public: // Action getters

    TsneSettingsAction& getTsneSettingsAction() { return _tsneSettingsAction; }
    ToggleAction& getUpdateStopAction() { return _updateStopAction; }
    ToggleAction& getFixScaleAction() { return _fixScaleAction; }
    ScaleDownUpActions& getScaleDownUpActions() { return _scaleUpDownActions; }
    IntegralAction& getVisRangeSlider() { return _visRangeAction; }
    IntegralAction& getVisBudgetMinSlider() { return _visBudgetMinAction; }
    IntegralAction& getVisBudgetMaxSlider() { return _visBudgetMaxAction; }
    IntegralAction& getVisBudgetTargetSlider() { return _visBudgetTargetAction; }
    ToggleAction& getRangeHeuristicToggle() { return _rangeHeuristicAction; }
    IntegralAction& getLandmarkFilterSlider() { return _landmarkFilterSlider; }
    ToggleAction& getLandmarkFilterToggle() { return _landmarkFilterToggle; }
    ColorMap2DAction& getColorMapRoiEmbAction() { return _colorMapRoiEmbAction; }
    ColorMap2DAction& getColorMapFirstEmbAction() { return _colorMapFirstEmbAction->getColorMapAction(); }
    ToggleAction& getRecolorDuringUpdates() { return _recolorDuringUpdates; }
    DecimalAction& getEmbScalingSlider() { return _embScalingSlider; }
    TriggerAction& getRecomputeScaleTrigger() { return _recomputeScale; }
    ToggleAction& getRandomInitMetaToggle() { return _randomInitMeta; }
    TriggerAction& getCompRepresentsTrigger() { return _compRepresents; }

public: // Getter

    IDMapping& getIDMap() { return _idMap; }
    const IDMapping& getIDMap() const { return _idMap; }

    TsneAnalysis& getTsneAnalysis() { return _tsneAnalysis;}

    utils::VisualBudgetRange getVisualBudgetRange() const;

    const Eigen::MatrixXui& getImageIndices() const {return _imageIndices; }

    // returns embedding extends after 100 iterations which are used for embedding rescaling

    utils::EmbeddingExtends getRefEmbExtends() const { return _refEmbExtends; }
    utils::EmbeddingExtends getCurrentEmbExtends() const { return _currentEmbExtends; }

    uint32_t getLandmarkFilterNumber() const { return _landmarkFilterToggle.isChecked() ? _landmarkFilterSlider.getValue() : 0; }

    bool getRoiGoodForUpdate() const { return _RoiGoodForUpdate; }

    std::pair<float, float> getEmdScalingFactors() const { return _embScaling; }

    const uint32_t getScaleLevel() const { return _currentScaleLevel;  }

public: // Setter

    /** Sets _currentScaleLevel and enable/disables UI buttons for going a scale up a down accordingly */
    void setScale(uint32_t scale);

    /** Sets _inputImageSize and copmutes _imageIndices*/
    void initImageSize(const QSize imgSize);

    /** layer Roi values are in image coordinates, view ROI are viewer size dependend. Only layer Roi is actually used, view Roi values are obsolete */
    void setROI(const utils::Vector2D layerRoiBottomLeft, const utils::Vector2D layerRoiTopRight, const utils::Vector2D viewRoiBottomLeft, const utils::Vector2D viewRoiTopRight);

    /** Sets _currentEmbExtends and _embScaling, updates UI text in _embScaleExtAndFac  */
    void setCurrentEmbExtends(utils::EmbeddingExtends extends);

    void setRefEmbExtends(utils::EmbeddingExtends extends);

    /** Set Min and Max visual values, range is determined from them */
    void setVisualBudgetRange(const uint32_t visBudgetMin, const uint32_t visBudgetMax);

    /** Set Min visual value, Max is determined from range, which is kept*/
    void setVisualBudgetRange(const uint32_t visBudgetMin);

protected:
    void setIDMap(const IDMapping& idMap) {
        _idMap = idMap;
    }

signals:
    void starttSNE(bool noExaggeration = true);
    void stoptSNE();

    void started();
    void finished();

    void noUpdate(NoUpdate reason);

    void setRoiInSequenceView(const utils::ROI& roi);
    void updateMetaData();

private slots:

    void starttSNEAnalysis();

    /** Interrupt the gradient descent */
    void stoptSNEAnalysis();

    void refineView();

    void coarsenView();
        
    void compRepresents();

    void updateEmbScaling();

    void computeUpdate(const utils::TraversalDirection direction = utils::TraversalDirection::AUTO);

    void publishSelectionData();

protected:
    TsneSettingsAction&     _tsneSettingsAction;    /** Reference to TSNE settings action from the HSNE analysis */
    HsneHierarchy&          _hsneHierarchy;         /** Reference to HSNE hierarchy */
    Dataset<Points>         _input;                 /** Input dataset reference */
    Dataset<Points>         _embedding;             /** Embedding dataset reference */
    Dataset<Points>         _pointInitTypes;        /** PointInitTypes dataset reference */
    Dataset<Points>         _regHsneTopLevel;       /** Top scale level embedding dataset reference */
    Dataset<Points>         _regTopLevelScatterCol; /** Top scale level embedding scatter colors dataset reference */
    Dataset<Points>         _roiRepresentation;     /** Roi representation dataset reference */
    Dataset<Points>         _numberTransitions;     /** Transitions number dataset reference */
    Dataset<Points>         _colorScatterRoiHSNE;   /** Color scatter ROI tSNE dataset reference */

    Dataset<Points>         _firstEmbedding;        /** Top scale level embedding dataset reference */
    Dataset<Points>         _topLevelLandmarkData;  /** Top scale level landmarks data dataset reference*/

protected: // UI elements
    ToggleAction            _updateStopAction;      /** Stop updating action */
    DecimalAction           _thresholdAction;       /** Set landmark influence treshold */
    ToggleAction            _influenceHeuristic;    /** Whether landmark influence heuristic or treshold is used */
    // TODO: introduce a log slider like e.g. this https://stackoverflow.com/a/68227820/16767931
    IntegralAction          _visRangeAction;        /** Visual Range */
    IntegralAction          _visBudgetMinAction;    /** Visual Budget minimum */
    IntegralAction          _visBudgetMaxAction;    /** Visual Budget maximum */
    IntegralAction          _visBudgetTargetAction; /** Visual Budget target*/
    ToggleAction            _rangeHeuristicAction;  /** Range heuristic action */
    StatusAction            _currentScaleAction;    /** Current Scale */
    ScaleDownUpActions      _scaleUpDownActions;    /** Scale Up and Down actions */
    ToggleAction            _fixScaleAction;        /** Fix Scale level action */
    IntegralAction          _landmarkFilterSlider;  /** Landmark Filter slider */
    ToggleAction            _landmarkFilterToggle;  /** Landmark Filter toggle */
    ColorMap2DAction        _colorMapRoiEmbAction;  /** Color map action */
    RecolorAction*          _colorMapFirstEmbAction;/** Color map action for first embedding */
    ToggleAction            _recolorDuringUpdates;  /** Toggles whether recoloring should happen only at the end of the gradient descent or continuously */
    DecimalAction           _embScalingSlider;      /** Embedding scaling factor multiplier slider */
    StatusAction            _embScaleFac;           /** Embedding scaling factor */
    StatusAction            _embCurrExt;            /** Embedding current extends */
    StatusAction            _embMaxExt;             /** Embedding max extends */
    ToggleAction            _noExaggerationUpdate;  /** Whether to set exageration to zero for each new embedding */
    TriggerAction           _recomputeScale;        /** Recompute Scale Embedding trigger */
    ToggleAction            _randomInitMeta;        /** Whether the random init should reset the init meta data */
    TriggerAction           _compRepresents;        /** compute representative landmarks on top scale */
    TriggerAction           _copySelectedAttributes;/** Copy selected attributes into data set */

private:
    uint32_t                _currentScaleLevel;     /** The scale the current embedding is a part of */

    float                   _tresh_influence;       /** Threshold to be used for selecting influencing and influenced points */

    QSize                   _inputImageSize;        /** Size (width and height) of the input dataset image */
    uint32_t                _numImagePoints;        /** Total number of points in image */
    Eigen::MatrixXui        _imageIndices;          /** 2D matrix of global image indices */

    IDMapping               _idMap;                 /** Maps global IDs (key) to their position in embedding (array) */

    utils::ROI              _roi;                   /** (0,0) is buttom left from user perspective, x-axis goes to the right */
    bool                    _RoiGoodForUpdate;      /** Lock that decides whether a scale update should be computed */
    bool                    _updateMetaDataset;     /** Lock that decides whether _pointInitTypes shoule updated, happens on first embedding update */

    InteractiveHsnePlugin*  _hsneAnalysisPlugin;    /** Pointer to HSNE analysis plugin */

    uint32_t                _visualRange;           /** Visual Range parameter, for now hardcoded */
    uint32_t                _visBudgetMax;          /** Visual Range parameter, for now hardcoded */
    bool                    _lockBudgetSlider;      /** Lock that blocks a budget slider to update  */

    utils::EmbeddingExtends _currentEmbExtends;
    std::pair<float, float> _embScaling;
    utils::EmbeddingExtends _refEmbExtends;

    std::vector<float>      _initEmbedding;

    HsneMatrix              _newTransitionMatrix;

    utils::CyclicLock       _updateRoiImageLock;

    HsneScaleUpdate         _hsneScaleUpdate;
    TsneAnalysis            _tsneAnalysis;          /** TSNE analysis */

};
