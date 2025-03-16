#include "HsneSettingsAction.h"
#include "InteractiveHSNEPlugin.h"

using namespace mv::gui;

HsneSettingsAction::HsneSettingsAction(InteractiveHsnePlugin* hsneAnalysisPlugin) :
    GroupAction(hsneAnalysisPlugin, "HsneSettingsAction", true),
    _hsneAnalysisPlugin(hsneAnalysisPlugin),
    _hsneParameters(),
    _generalHsneSettingsAction(*this),
    _advancedHsneSettingsAction(*this),
    _interactiveScaleAction(this, _hsneAnalysisPlugin, _tsneSettingsAction, hsneAnalysisPlugin->getHierarchy(), 
        hsneAnalysisPlugin->getInputDataset<Points>(),  hsneAnalysisPlugin->getOutputDataset<Points>(), 
        hsneAnalysisPlugin->getFirstEmbeddingDataset(), hsneAnalysisPlugin->getTopLevelLandmarkDataDataset(),
        hsneAnalysisPlugin->getPointInitTypesDataset(), hsneAnalysisPlugin->getRoiRepresentationDataset(), 
        hsneAnalysisPlugin->getNumberTransitionsDataset(), hsneAnalysisPlugin->getColorScatterRoiHSNEDataset(),
        hsneAnalysisPlugin->getRegHsneTopLevelDataset()),
    _tsneSettingsAction(this),
    _viewportSequenceAction(this),
    _meanShiftAction(this, hsneAnalysisPlugin->getFirstEmbeddingDataset(), hsneAnalysisPlugin->getTopLevelEmbClustersDataset()),
    _dimensionSelectionAction(this)
//    _hsneImageViewportSharingAction(this) // [REMOVE]
{
    setText("HSNE");
    setObjectName("Settings");

    _tsneSettingsAction.setObjectName("TSNE");

    const auto updateReadOnly = [this]() -> void {
        _generalHsneSettingsAction.setReadOnly(isReadOnly());
        _advancedHsneSettingsAction.setReadOnly(isReadOnly());
        _interactiveScaleAction.setReadOnly(isReadOnly());
        _tsneSettingsAction.setReadOnly(isReadOnly());
    };

    connect(this, &GroupAction::readOnlyChanged, this, [this, updateReadOnly](const bool& readOnly) {
        updateReadOnly();
    });

    updateReadOnly();
}

HsneParameters& HsneSettingsAction::getHsneParameters()
{
    return _hsneParameters;
}

TsneParameters& HsneSettingsAction::getTsneParameters()
{
    return _tsneSettingsAction.getTsneParameters();
}
