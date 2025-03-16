#pragma once

#include "HsneParameters.h"
#include "TsneParameters.h"
#include "GeneralHsneSettingsAction.h"
#include "AdvancedHsneSettingsAction.h"
#include "HsneScaleAction.h"
#include "TsneSettingsAction.h"
#include "ViewportSequence.h"
#include "MeanShiftAction.h"
#include "DimensionSelectionAction.h"
#include "ViewportSharingActions.h"

using namespace mv::gui;

class QMenu;
class InteractiveHsnePlugin;

/**
 * HSNE setting action class
 *
 * Action class for HSNE settings
 *
 * @author Thomas Kroes
 */
class HsneSettingsAction : public GroupAction
{
public:

    /**
     * Constructor
     * @param hsneAnalysisPlugin Pointer to HSNE analysis plugin
     */
    HsneSettingsAction(InteractiveHsnePlugin* hsneAnalysisPlugin);

    /** Get HSNE/TSNE parameters */
    HsneParameters& getHsneParameters();
    TsneParameters& getTsneParameters();

public: // Action getters

    GeneralHsneSettingsAction& getGeneralHsneSettingsAction() { return _generalHsneSettingsAction; }
    AdvancedHsneSettingsAction& getAdvancedHsneSettingsAction() { return _advancedHsneSettingsAction; }
    HsneScaleAction& getInteractiveScaleAction() { return _interactiveScaleAction; }
    TsneSettingsAction& getTsneSettingsAction() { return _tsneSettingsAction; }
    ViewportSequence& getViewportSequenceAction() { return _viewportSequenceAction; }
    MeanShiftAction& getMeanShiftActionAction() { return _meanShiftAction; }
    DimensionSelectionAction& getDimensionSelectionAction() { return _dimensionSelectionAction; }

    // [REMOVE]
    //ViewportSharingActions& getHsneImageViewportSharingAction() { return _hsneImageViewportSharingAction; }

protected:
    InteractiveHsnePlugin*          _hsneAnalysisPlugin;            /** Pointer to HSNE analysis plugin */
    HsneParameters                  _hsneParameters;                /** HSNE parameters */
    GeneralHsneSettingsAction       _generalHsneSettingsAction;     /** General HSNE settings action */
    AdvancedHsneSettingsAction      _advancedHsneSettingsAction;    /** Advanced HSNE settings action */
    HsneScaleAction                 _interactiveScaleAction;        /** Interactive scale action */
    TsneSettingsAction              _tsneSettingsAction;            /** TSNE settings action */
    ViewportSequence                _viewportSequenceAction;        /** Viewport sequence action */
    MeanShiftAction                 _meanShiftAction;               /** Mean shift top level embedding action */
    DimensionSelectionAction        _dimensionSelectionAction;      /** Dimension selection action */

    // [REMOVE]
    //ViewportSharingActions  _hsneImageViewportSharingAction;        /** Viewport sharing action */

    friend class AdvancedHsneSettingsAction;
};
