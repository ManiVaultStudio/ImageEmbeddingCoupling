#pragma once

#include "actions/ToggleAction.h"
#include "actions/TriggerAction.h"
#include "actions/DecimalAction.h"
#include "actions/ColorMapAction.h"
#include "PointData/PointData.h"

#include "TsneAnalysis.h"
#include "RecolorAction.h"

using namespace mv;
using namespace mv::gui;
using namespace mv::util;

class QMenu;

class InteractiveHsnePlugin;
class HsneHierarchy;
class TsneSettingsAction;

/**
 * HSNE scale action class
 *
 * Action class for HSNE scale
 *
 * @author Thomas Kroes
 */
class RegularHsneAction : public GroupAction
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
     */
    RegularHsneAction(QObject* parent, TsneSettingsAction& tsneSettingsAction, HsneHierarchy& hsneHierarchy, Dataset<Points> inputDataset, 
        Dataset<Points> embeddingDataset, Dataset<Points> embeddingScatColors, InteractiveHsnePlugin* hsneAnalysisPlugin);

    /**
     * Get the context menu for the action
     * @param parent Parent widget
     * @return Context menu
     */
    QMenu* getContextMenu(QWidget* parent = nullptr) override;

protected:

    /** Refine the landmarks based on the current selection */
    void refine();

public: // Action getters

    TsneSettingsAction& getTsneSettingsAction() { return _tsneSettingsAction; }
    TriggerAction& getRefineAction() { return _refineAction; }
    DecimalAction& getRefineThreshAction() { return _refineTresh; }
    ToggleAction& getRefineHeuristicAction() { return _refineHeuristic; }
    RecolorAction& getColorMapActionAction() { return _recolorAction; }

    void setScale(unsigned int scale)
    {
        _currentScaleLevel = scale;
    }

    void setDrillIndices(const std::vector<uint32_t>& drillIndices)
    {
        _drillIndices = drillIndices;
        _isTopScale = false;
    }

protected:
    TsneSettingsAction&     _tsneSettingsAction;    /** Reference to TSNE settings action from the HSNE analysis */
    TsneAnalysis            _tsneAnalysis;          /** TSNE analysis */
    HsneHierarchy&          _hsneHierarchy;         /** Reference to HSNE hierarchy */
    Dataset<Points>         _input;                 /** Input dataset reference */
    Dataset<Points>         _embedding;             /** Embedding dataset reference */
    Dataset<Points>         _embeddingScatColors;   /** Embedding scatter colors dataset reference */
    Dataset<Points>         _refineEmbedding;       /** Refine embedding dataset reference */
    Dataset<Points>         _refineEmbScatColors;   /** Refine embedding scatter colors dataset reference */
    ToggleAction            _refineHeuristic;       /** Use refine heuristic toggle */
    DecimalAction           _refineTresh;           /** Threshold value for refinement */
    TriggerAction           _refineAction;          /** Refine action */
    RecolorAction           _recolorAction;         /** Color map action for top HSNE level */

private:
    std::vector<uint32_t>   _drillIndices;          /** Vector relating local indices to scale relative indices */
    bool                    _isTopScale;            /** Whether current scale is the top scale */
    unsigned int            _currentScaleLevel;     /** The scale the current embedding is a part of */

    InteractiveHsnePlugin*  _hsneAnalysisPlugin;    /** Pointer to HSNE analysis plugin */
    HsneMatrix              _newTransitionMatrix;   /** Transition matrix */

};
