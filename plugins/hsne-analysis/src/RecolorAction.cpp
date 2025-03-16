#include "RecolorAction.h"

#include <QMenu>

QMenu* RecolorAction::getContextMenu(QWidget* parent)
{
    auto menu = new QMenu(text(), parent);
    menu->addAction(&_colorMapAction);
    return menu;
}
