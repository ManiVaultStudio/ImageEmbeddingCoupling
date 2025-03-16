#include "AdvancedHsneSettingsAction.h"
#include "HsneSettingsAction.h"
#include "InteractiveHsnePlugin.h"

using namespace mv::gui;

AdvancedHsneSettingsAction::AdvancedHsneSettingsAction(HsneSettingsAction& hsneSettingsAction) :
    GroupAction(&hsneSettingsAction, "AdvancedHsneSettingsAction"),
    _hsneSettingsAction(hsneSettingsAction),
    _numWalksForLandmarkSelectionAction(this, "#walks for landmark sel."),
    _numWalksForLandmarkSelectionThresholdAction(this, "#thres for landmark sel."),
    _randomWalkLengthAction(this, "Random walk length"),
    _numWalksForAreaOfInfluenceAction(this, "#walks for aoi"),
    _minWalksRequiredAction(this, "Minimum #walks required"),
    _numTreesAknnAction(this, "KNN trees (Annoy)"),
    _HNSW_M_Action(this, "KNN M (HNSW)"),
    _HNSW_eff_Action(this, "KNN ef (HNSW)"),
    _useOutOfCoreComputationAction(this, "Out-of-core computation"),
    _initWithPCAAction(this, "Init with PCA (of landmark data)"),
    _pcaAlgorithmAction(this, "PCA alg"),
    _hardCutOffAction(this, "Hard cut off"),
    _hardCutOffPercentageAction(this, "% hard cut off")
{
    setText("Advanced HSNE");
    setObjectName("Advanced HSNE");

    /// UI set up: add actions
    for (auto& action : WidgetActions{ &_numWalksForLandmarkSelectionAction, &_numWalksForLandmarkSelectionThresholdAction,
        &_randomWalkLengthAction, &_numWalksForAreaOfInfluenceAction, &_minWalksRequiredAction,
        &_minWalksRequiredAction, &_numTreesAknnAction, &_HNSW_M_Action, &_HNSW_eff_Action, &_useOutOfCoreComputationAction,
        &_initWithPCAAction, &_hardCutOffAction, &_hardCutOffPercentageAction })
        addAction(action);

    _numWalksForLandmarkSelectionAction.setDefaultWidgetFlags(IntegralAction::SpinBox);
    _numWalksForLandmarkSelectionThresholdAction.setDefaultWidgetFlags(IntegralAction::SpinBox);
    _randomWalkLengthAction.setDefaultWidgetFlags(IntegralAction::SpinBox);
    _numWalksForAreaOfInfluenceAction.setDefaultWidgetFlags(IntegralAction::SpinBox);
    _minWalksRequiredAction.setDefaultWidgetFlags(IntegralAction::SpinBox);
    _numTreesAknnAction.setDefaultWidgetFlags(IntegralAction::SpinBox | IntegralAction::Slider);
    _HNSW_M_Action.setDefaultWidgetFlags(IntegralAction::SpinBox | IntegralAction::Slider);
    _HNSW_eff_Action.setDefaultWidgetFlags(IntegralAction::SpinBox | IntegralAction::Slider);
    _useOutOfCoreComputationAction.setDefaultWidgetFlags(ToggleAction::CheckBox);
    _initWithPCAAction.setDefaultWidgetFlags(ToggleAction::CheckBox);
    _hardCutOffAction.setDefaultWidgetFlags(ToggleAction::CheckBox);
    _hardCutOffPercentageAction.setDefaultWidgetFlags(IntegralAction::SpinBox | IntegralAction::Slider);

    _numWalksForLandmarkSelectionAction.setToolTip("Number of walks for landmark selection");
    _numWalksForLandmarkSelectionThresholdAction.setToolTip("Threshold for landmark selection");
    _randomWalkLengthAction.setToolTip("Number of walks for landmark selection threshold");
    _numWalksForAreaOfInfluenceAction.setToolTip("Number of walks for area of influence");
    _minWalksRequiredAction.setToolTip("Minimum number of walks required");
    _numTreesAknnAction.setToolTip("Number of KNN trees (Annoy). More trees gives higher precision.");
    _HNSW_M_Action.setToolTip("HNSW parameter M. Higher values work better on datasets with high intrinsic dimensionality and/or high recall, while lower values work better for datasets with low intrinsic dimensionality and/or low recalls. The range M=12-48 is ok for the most of the use cases.");
    _HNSW_eff_Action.setToolTip("HNSW parameter M ef and ef_construction. Higher values lead to more accurate but slower construction and search. ef cannot be set lower than the number of queried nearest neighbors k. The value ef of can be anything between k and the size of the dataset.");
    _useOutOfCoreComputationAction.setToolTip("Use out-of-core computation");
    _initWithPCAAction.setToolTip("Init embedding with PCA (of top level landmark data)");
    _pcaAlgorithmAction.setToolTip("Type of PCA algorithm");
    _hardCutOffAction.setToolTip("Select landmarks based on a user provided hard percentage cut off, instead of data-driven");
    _hardCutOffPercentageAction.setToolTip("Percentage of previous level landmarks to use in next level when using the hard cut off");

    const auto& hsneParameters = hsneSettingsAction.getHsneParameters();

    _numWalksForLandmarkSelectionAction.initialize(1, 1000, hsneParameters.getNumWalksForLandmarkSelection());
    _numWalksForLandmarkSelectionThresholdAction.initialize(0, 10, hsneParameters.getNumWalksForLandmarkSelectionThreshold(), 3);
    _randomWalkLengthAction.initialize(1, 100, hsneParameters.getRandomWalkLength());
    _numWalksForAreaOfInfluenceAction.initialize(1, 500, hsneParameters.getNumWalksForAreaOfInfluence());
    _minWalksRequiredAction.initialize(0, 100, hsneParameters.getMinWalksRequired());
    _numTreesAknnAction.initialize(1, 1024, hsneParameters.getNumTreesAKNN());
    _HNSW_M_Action.initialize(1, 1024, hsneParameters.getHNSW_M());
    _HNSW_eff_Action.initialize(1, 1024, hsneParameters.getHNSW_eff());
    _useOutOfCoreComputationAction.setChecked(hsneParameters.useOutOfCoreComputation());
    _initWithPCAAction.setChecked(true);
    _pcaAlgorithmAction.initialize(QStringList({ "SVD", "COV" }), "COV");
    _hardCutOffAction.setChecked(true);
    _hardCutOffPercentageAction.initialize(0, 1, 0.25f,3);
    _hardCutOffPercentageAction.setSingleStep(0.01f);


    const auto updateNumWalksForLandmarkSelectionAction = [this]() -> void {
        _hsneSettingsAction.getHsneParameters().setNumWalksForLandmarkSelection(_numWalksForLandmarkSelectionAction.getValue());
    };

    const auto updateNumWalksForLandmarkSelectionThreshold = [this]() -> void {
        _hsneSettingsAction.getHsneParameters().setNumWalksForLandmarkSelectionThreshold(_numWalksForLandmarkSelectionThresholdAction.getValue());
    };

    const auto updateRandomWalkLength = [this]() -> void {
        _hsneSettingsAction.getHsneParameters().setRandomWalkLength(_randomWalkLengthAction.getValue());
    };

    const auto updateNumWalksForAreaOfInfluence = [this]() -> void {
        _hsneSettingsAction.getHsneParameters().setNumWalksForAreaOfInfluence(_numWalksForAreaOfInfluenceAction.getValue());
    };

    const auto updateMinWalksRequired = [this]() -> void {
        _hsneSettingsAction.getHsneParameters().setMinWalksRequired(_minWalksRequiredAction.getValue());
    };

    // Annoy parameter
    const auto updateNumTreesAknn = [this]() -> void {
        _hsneSettingsAction.getHsneParameters().setNumTreesAKNN(_numTreesAknnAction.getValue());
        _hsneSettingsAction.getTsneParameters().setNumTrees(_numTreesAknnAction.getValue());
    };

    // HSNW parameter
    const auto updateHSNWM = [this]() -> void {
        _hsneSettingsAction.getHsneParameters().setHNSW_M(_HNSW_M_Action.getValue());
        _hsneSettingsAction.getTsneParameters().setHNSW_M(_HNSW_M_Action.getValue());
    };

    // HSNW parameter
    const auto updateHSNWeff = [this]() -> void {
        _hsneSettingsAction.getHsneParameters().setHNSW_eff(_HNSW_eff_Action.getValue());
        _hsneSettingsAction.getTsneParameters().setHNSW_eff(_HNSW_eff_Action.getValue());
    };

    const auto updateUseOutOfCoreComputation = [this]() -> void {
        _hsneSettingsAction.getHsneParameters().useOutOfCoreComputation(_useOutOfCoreComputationAction.isChecked());
    };

    const auto updateHardCutOff = [this]() -> void {
        _hsneSettingsAction.getHsneParameters().setHardCutOff(_hardCutOffAction.isChecked());
    };

    const auto updateHardCutOffPercetage = [this](uint32_t target = 10'000) -> void {
        _hsneSettingsAction.getHsneParameters().setHardCutOffPercentage(_hardCutOffPercentageAction.getValue());

        // set new scale value in UI if hardCutOff is checked
        auto numScales = _hsneSettingsAction._hsneAnalysisPlugin->compNumHierarchyScales();
        
        if (&_hsneSettingsAction._hsneAnalysisPlugin->getHsneSettingsAction() == nullptr)
            return;

        auto& scalesAction = _hsneSettingsAction._hsneAnalysisPlugin->getHsneSettingsAction().getGeneralHsneSettingsAction().getNumScalesAction();

        if (scalesAction.getValue() != numScales)
            scalesAction.setValue(numScales);

    };

    const auto updateInitWithPCA = [this]() -> void {
        _hsneSettingsAction.getHsneParameters().initWithPCA(_initWithPCAAction.isChecked());
    };

    const auto updateReadOnly = [this]() -> void {
        const auto enabled = !isReadOnly();

        _numWalksForLandmarkSelectionAction.setEnabled(enabled);
        _numWalksForLandmarkSelectionThresholdAction.setEnabled(enabled);
        _randomWalkLengthAction.setEnabled(enabled);
        _numWalksForAreaOfInfluenceAction.setEnabled(enabled);
        _minWalksRequiredAction.setEnabled(enabled);
        _numTreesAknnAction.setEnabled(enabled);
        _HNSW_M_Action.setEnabled(enabled);
        _HNSW_eff_Action.setEnabled(enabled);
        _useOutOfCoreComputationAction.setEnabled(enabled);
        _initWithPCAAction.setEnabled(enabled);
        _pcaAlgorithmAction.setEnabled(enabled);
        _hardCutOffAction.setEnabled(enabled);
        _hardCutOffPercentageAction.setEnabled(enabled);
    };

    connect(&_numWalksForLandmarkSelectionAction, &IntegralAction::valueChanged, this, [this, updateNumWalksForLandmarkSelectionAction]() {
        updateNumWalksForLandmarkSelectionAction();
    });

    connect(&_numWalksForLandmarkSelectionThresholdAction, &DecimalAction::valueChanged, this, [this, updateNumWalksForLandmarkSelectionThreshold]() {
        updateNumWalksForLandmarkSelectionThreshold();
    });

    connect(&_randomWalkLengthAction, &IntegralAction::valueChanged, this, [this, updateRandomWalkLength]() {
        updateRandomWalkLength();
    });

    connect(&_numWalksForAreaOfInfluenceAction, &IntegralAction::valueChanged, this, [this, updateNumWalksForAreaOfInfluence]() {
        updateNumWalksForAreaOfInfluence();
    });

    connect(&_minWalksRequiredAction, &IntegralAction::valueChanged, this, [this, updateMinWalksRequired]() {
        updateMinWalksRequired();
    });

    connect(&_HNSW_M_Action, &IntegralAction::valueChanged, this, [this, updateNumTreesAknn]() {
        updateNumTreesAknn();
    });

    connect(&_HNSW_eff_Action, &IntegralAction::valueChanged, this, [this, updateNumTreesAknn]() {
        updateNumTreesAknn();
    });

    connect(&_numTreesAknnAction, &IntegralAction::valueChanged, this, [this, updateNumTreesAknn]() {
        updateNumTreesAknn();
    });

    connect(&_useOutOfCoreComputationAction, &ToggleAction::toggled, this, [this, updateUseOutOfCoreComputation]() {
        updateUseOutOfCoreComputation();
    });

    connect(&_initWithPCAAction, &ToggleAction::toggled, this, [this, updateInitWithPCA]() {
        updateInitWithPCA();
    });

    connect(&_hardCutOffAction, &ToggleAction::toggled, this, [this, updateHardCutOff]() {
        updateHardCutOff();
    });

    connect(&_hardCutOffPercentageAction, &DecimalAction::valueChanged, this, [this, updateHardCutOffPercetage]() {
        updateHardCutOffPercetage();
    });

    connect(this, &GroupAction::readOnlyChanged, this, [this, updateReadOnly](const bool& readOnly) {
        updateReadOnly();
    });

    updateNumWalksForLandmarkSelectionAction();
    updateNumWalksForLandmarkSelectionThreshold();
    updateRandomWalkLength();
    updateNumWalksForAreaOfInfluence();
    updateMinWalksRequired();
    updateNumTreesAknn();
    updateHSNWM();
    updateHSNWeff();
    updateUseOutOfCoreComputation();
    updateHardCutOff();
    updateHardCutOffPercetage();
    updateReadOnly();
}
