#include "TsneComputationAction.h"
#include "TsneSettingsAction.h"

#include <QHBoxLayout>
#include <QMenu>

using namespace mv::gui;

TsneComputationAction::TsneComputationAction(QObject* parent) :
    HorizontalGroupAction(parent, "TsneComputationAction"),
    _continueComputationAction(this, "Continue"),
    _stopComputationAction(this, "Stop")
{
    setText("Computation");

    addAction(&_continueComputationAction);
    addAction(&_stopComputationAction);

    _continueComputationAction.setToolTip("Continue with the tSNE computation");
    _stopComputationAction.setToolTip("Stop the current tSNE computation");
}

QMenu* TsneComputationAction::getContextMenu(QWidget* parent /*= nullptr*/)
{
    auto menu = new QMenu(text(), parent);

    menu->addAction(&_continueComputationAction);
    menu->addAction(&_stopComputationAction);

    return menu;
}

