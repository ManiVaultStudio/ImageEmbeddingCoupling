#include "TsneSettingsAction.h"

#include <QMenu>

using namespace mv::gui;

TsneSettingsAction::TsneSettingsAction(QObject* parent) :
    GroupAction(parent, "TsneSettingsAction", false),
    _tsneParameters(),
    _generalTsneSettingsAction(*this)
{
    setText("TSNE");

    const auto updateReadOnly = [this]() -> void {
        _generalTsneSettingsAction.setReadOnly(isReadOnly());
    };

    connect(this, &GroupAction::readOnlyChanged, this, [this, updateReadOnly](const bool& readOnly) {
        updateReadOnly();
    });

    updateReadOnly();
}

QMenu* TsneSettingsAction::getContextMenu(QWidget* parent /*= nullptr*/)
{
    auto menu = new QMenu(text(), parent);

    auto& computationAction = _generalTsneSettingsAction.getComputationAction();

    menu->addAction(&computationAction.getContinueComputationAction());
    menu->addAction(&computationAction.getStopComputationAction());

    return menu;
}
