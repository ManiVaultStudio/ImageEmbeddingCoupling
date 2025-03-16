#include "GeneralTsneSettingsAction.h"
#include "TsneSettingsAction.h"

#include <QLabel>
#include <QPushButton>
#include <QGridLayout>

using namespace mv::gui;

GeneralTsneSettingsAction::GeneralTsneSettingsAction(TsneSettingsAction& tsneSettingsAction) :
    GroupAction(&tsneSettingsAction, "GeneralTsneSettingsAction", true),
    _tsneSettingsAction(tsneSettingsAction),
    _numNewIterationsAction(this, "New iterations"),
    _datasetSelectionAction(this, "Data set"),
    _numDefaultUpdateIterationsAction(this, "Default iterations"),
    _exaggerationIterAction(this, "Exaggeration Iterations"),
    _exponentialDecayAction(this, "Exponential decay"),
    _exaggerationFactorAction(this, "Exaggeration facor"),
    _exaggerationToggleAction(this, "Auto exaggeration"),
    _iterationsPushlishExtendAction(this, "Set Ref. extends at"),
    _publishExtendsOnceAction(this, "Set Ref. extends once", true),
    _numComputatedIterationsAction(this, "Computed iterations"),
    _computationAction(this),
    _embDatasets()
{
    setText("TSNE");
    setObjectName("General TSNE");

    /// UI set up: add actions
    for (auto& action : WidgetActions{ &_datasetSelectionAction, &_exaggerationIterAction, &_exponentialDecayAction,
        & _exaggerationFactorAction, & _exaggerationToggleAction, & _iterationsPushlishExtendAction, & _publishExtendsOnceAction,
        & _numNewIterationsAction, & _numDefaultUpdateIterationsAction, & _numComputatedIterationsAction, & _computationAction })
        addAction(action);

    _datasetSelectionAction.setDefaultWidgetFlags(OptionAction::ComboBox);
    _numNewIterationsAction.setDefaultWidgetFlags(IntegralAction::SpinBox);
    _exaggerationIterAction.setDefaultWidgetFlags(IntegralAction::SpinBox);
    _exponentialDecayAction.setDefaultWidgetFlags(IntegralAction::SpinBox);
    _numComputatedIterationsAction.setDefaultWidgetFlags(IntegralAction::LineEdit);
    _numNewIterationsAction.setDefaultWidgetFlags(IntegralAction::SpinBox);
    _numDefaultUpdateIterationsAction.setDefaultWidgetFlags(IntegralAction::SpinBox);
    _iterationsPushlishExtendAction.setDefaultWidgetFlags(IntegralAction::SpinBox);

    _datasetSelectionAction.initialize();
    _numDefaultUpdateIterationsAction.initialize(0, 10000, 2000u);
    _numNewIterationsAction.initialize(0, 10000, 0);
    _iterationsPushlishExtendAction.initialize(1, 10000, 250);
    _exaggerationIterAction.initialize(1, 10000, 250);
    _exponentialDecayAction.initialize(1, 10000, 70);
    _exaggerationFactorAction.initialize(0, 100, 4, 2);

    _numComputatedIterationsAction.initialize(0, 100000, 0);
    _numComputatedIterationsAction.setEnabled(false);

    _exaggerationToggleAction.setChecked(true);
    _exaggerationToggleAction.setToolTip("Auto val is: 4 + (number of embedded points) / 60000.0");
    _exaggerationFactorAction.setEnabled(false);

    _iterationsPushlishExtendAction.setToolTip("Should be larger or equal to number of exaggeration iterations");
    _publishExtendsOnceAction.setToolTip("Only set the reference extends once, when computing the top level embedding first");

    const auto updateNumIterations = [this]() -> void {
        _tsneSettingsAction.getTsneParameters().setNumIterations(_numDefaultUpdateIterationsAction.getValue());
    };

    const auto updateIterationsPushlishExtend = [this]() -> void {
        if ((_numComputatedIterationsAction.getValue() > 0) && _publishExtendsOnceAction.isChecked())
            return;

        _tsneSettingsAction.getTsneParameters().setPublishExtendsAtIteration(_iterationsPushlishExtendAction.getValue());
    };

    const auto updateExaggerationIter = [this]() -> void {
        _tsneSettingsAction.getTsneParameters().setExaggerationIter(_exaggerationIterAction.getValue());
    };

    const auto updateExponentialDecay = [this]() -> void {
        _tsneSettingsAction.getTsneParameters().setExponentialDecayIter(_exponentialDecayAction.getValue());
    };

    const auto updateExaggerationFactor = [this]() -> void {
        double exaggeration = -1.0; // auto exaggeration is computed in TsneAnalysis.cpp based on number of landmarks in scale

        if (_exaggerationToggleAction.isChecked())
            _exaggerationFactorAction.setValue(0);
        else
            exaggeration = _exaggerationFactorAction.getValue();

        _tsneSettingsAction.getTsneParameters().setExaggerationFactor(exaggeration);
    };

    const auto updateReadOnly = [this]() -> void {
        auto enable = !isReadOnly();

        _numNewIterationsAction.setEnabled(enable);
        _numDefaultUpdateIterationsAction.setEnabled(enable);
        _iterationsPushlishExtendAction.setEnabled(enable);
        _publishExtendsOnceAction.setEnabled(enable);
        _exaggerationIterAction.setEnabled(enable);
        _exaggerationFactorAction.setEnabled(enable);
        _exaggerationToggleAction.setEnabled(enable);
        _exponentialDecayAction.setEnabled(enable);

        if ((_numComputatedIterationsAction.getValue() > 0) && _publishExtendsOnceAction.isChecked())
            _iterationsPushlishExtendAction.setEnabled(false);

        if (_numNewIterationsAction.getValue() == 0)
            enable = false;

        _computationAction.setEnabled(enable);
    };

    connect(&_numNewIterationsAction, &IntegralAction::valueChanged, this, [this](const std::int32_t& value) {
        _computationAction.setEnabled(true);

        if (value == 0)
            _computationAction.setEnabled(false);
    });

    connect(&_numDefaultUpdateIterationsAction, &IntegralAction::valueChanged, this, [this, updateNumIterations](const std::int32_t& value) {           
        updateNumIterations();
    });

    connect(&_numComputatedIterationsAction, &IntegralAction::valueChanged, this, [this](const std::int32_t& value) {

        if (_publishExtendsOnceAction.isChecked())
        {
            _iterationsPushlishExtendAction.setEnabled(false);
            _tsneSettingsAction.getTsneParameters().setPublishExtendsAtIteration(0);
        }
    });

    connect(&_publishExtendsOnceAction, &ToggleAction::toggled, this, [this, updateIterationsPushlishExtend](const bool& val) {

        if (_numComputatedIterationsAction.getValue() == 0)
            return;

        _iterationsPushlishExtendAction.setEnabled(!_publishExtendsOnceAction.isChecked());

        if (_publishExtendsOnceAction.isChecked())
            _tsneSettingsAction.getTsneParameters().setPublishExtendsAtIteration(0);
        else
            updateIterationsPushlishExtend();
        
        });

    connect(&_iterationsPushlishExtendAction, &IntegralAction::valueChanged, this, [this, updateIterationsPushlishExtend](const std::int32_t& value) {
        updateIterationsPushlishExtend();
        });

    connect(&_exaggerationIterAction, &IntegralAction::valueChanged, this, [this, updateExaggerationIter](const std::int32_t& value) {
        updateExaggerationIter();
        });

    connect(&_exponentialDecayAction, &IntegralAction::valueChanged, this, [this, updateExponentialDecay](const std::int32_t& value) {
        updateExponentialDecay();
        });

    connect(&_exaggerationFactorAction, &DecimalAction::valueChanged, this, [this, updateExaggerationFactor](const float& value) {
        updateExaggerationFactor();
        });

    connect(&_exaggerationToggleAction, &IntegralAction::toggled, this, [this, updateExaggerationFactor](const bool toggled) {
        _exaggerationFactorAction.setEnabled(!toggled);
        updateExaggerationFactor();
        });

    connect(this, &GroupAction::readOnlyChanged, this, [this, updateReadOnly](const bool& readOnly) {
        updateReadOnly();
    });

    updateNumIterations();
    updateIterationsPushlishExtend();
    updateExaggerationIter();
    updateExponentialDecay();
    updateExaggerationFactor();
    updateReadOnly();
    _numComputatedIterationsAction.setEnabled(false);

}

void GeneralTsneSettingsAction::setEmbDatasets(QMap<QString, QString>& embDatasets) 
{ 
    _embDatasets = embDatasets;

    _datasetSelectionAction.setOptions(_embDatasets.keys());
    _datasetSelectionAction.setCurrentIndex(0);
};

QString GeneralTsneSettingsAction::getCurrentEmbDataset() const
{
    return _embDatasets.value(_datasetSelectionAction.getCurrentText());
}
