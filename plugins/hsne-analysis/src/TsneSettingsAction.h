#pragma once

#include "TsneParameters.h"
#include "GeneralTsneSettingsAction.h"
#include "DimensionSelectionAction.h"

using namespace mv::gui;

class QMenu;

class TsneComputationAction;

/**
 * TSNE settings class
 *
 * Settings actions class for general/advanced HSNE/TSNE settings
 *
 * @author Thomas Kroes
 */
class TsneSettingsAction : public GroupAction
{
public:

    /**
     * Constructor
     * @param parent Pointer to parent object
     */
    TsneSettingsAction(QObject* parent);

    /**
     * Get the context menu for the action
     * @param parent Parent widget
     * @return Context menu
     */
    QMenu* getContextMenu(QWidget* parent = nullptr) override;

public: // Action getters

    TsneParameters& getTsneParameters() { return _tsneParameters; }
    GeneralTsneSettingsAction& getGeneralTsneSettingsAction() { return _generalTsneSettingsAction; }
    TsneComputationAction& getComputationAction() { return _generalTsneSettingsAction.getComputationAction(); }

protected:
    TsneParameters                  _tsneParameters;                /** TSNE parameters */
    GeneralTsneSettingsAction       _generalTsneSettingsAction;     /** General tSNE settings action */
};