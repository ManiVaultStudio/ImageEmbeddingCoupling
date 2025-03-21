#pragma once

#include "actions/IntegralAction.h"
#include "actions/ToggleAction.h"
#include "actions/OptionAction.h"
#include "actions/TriggerAction.h"
#include "actions/DecimalAction.h"

#include "TsneComputationAction.h"

using namespace mv::gui;

class QMenu;
class TsneSettingsAction;

/**
 * General TSNE setting action class
 *
 * Actions class for general TSNE settings
 *
 * @author Thomas Kroes
 */
class GeneralTsneSettingsAction : public GroupAction
{
public:

    /**
     * Constructor
     * @param tsneSettingsAction Reference to TSNE settings action
     */
    GeneralTsneSettingsAction(TsneSettingsAction& tsneSettingsAction);

public: // Action getters

    TsneSettingsAction& getTsneSettingsAction() { return _tsneSettingsAction; };
    OptionAction& getDatasetSelectionActionAction() { return _datasetSelectionAction; };
    IntegralAction& getExaggerationIterAction() { return _exaggerationIterAction; };
    IntegralAction& getExponentialDecayAction() { return _exponentialDecayAction; };
    DecimalAction& getExaggerationFactorAction() { return _exaggerationFactorAction; };
    ToggleAction& getExaggerationToggleAction() { return _exaggerationToggleAction; };
    IntegralAction& getIterationsPushlishExtendAction() { return _iterationsPushlishExtendAction; };
    ToggleAction& getPublishExtendsOnceAction() { return _publishExtendsOnceAction; };
    IntegralAction& getNumNewIterationsAction() { return _numNewIterationsAction; };
    IntegralAction& getNumDefaultUpdateIterationsAction() { return _numDefaultUpdateIterationsAction; };
    IntegralAction& getNumComputatedIterationsAction() { return _numComputatedIterationsAction; };
    TsneComputationAction& getComputationAction() { return _computationAction; }

public: // EmbDatasets
    void setEmbDatasets(QMap<QString, QString>& embDatasets);
    QString getCurrentEmbDataset() const;

protected:
    TsneSettingsAction&     _tsneSettingsAction;                    /** Reference to parent tSNE settings action */
    OptionAction            _datasetSelectionAction;                /** Which t-SNE dataset/analysis to work with action */
    IntegralAction          _exaggerationIterAction;                /** Exaggeration iteration action */
    IntegralAction          _exponentialDecayAction;                /** Exponential decay of exaggeration action */
    DecimalAction           _exaggerationFactorAction;              /** Exaggeration factor action */
    ToggleAction            _exaggerationToggleAction;              /** Exaggeration toggle action */
    IntegralAction          _iterationsPushlishExtendAction;        /** Number of iterations at which to publsh reference extends action */
    ToggleAction            _publishExtendsOnceAction;              /** Whether reference extends should only be set once, when the top level is computed */
    IntegralAction          _numNewIterationsAction;                /** Number of new iterations action */
    IntegralAction          _numDefaultUpdateIterationsAction;      /** Number of default update iterations action */
    IntegralAction          _numComputatedIterationsAction;         /** Number of computed iterations action */
    TsneComputationAction   _computationAction;                     /** Computation action */

private:
    QMap<QString, QString>  _embDatasets;
};
