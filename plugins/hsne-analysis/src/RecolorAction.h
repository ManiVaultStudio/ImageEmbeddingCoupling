#pragma once

#include "actions/GroupAction.h"
#include "actions/ColorMap2DAction.h"

using namespace mv::gui;
using namespace mv::util;

class QMenu;

/// /////////////// ///
/// Single Recolor  ///
/// /////////////// ///
class RecolorAction : public GroupAction
{
    Q_OBJECT
public:
    RecolorAction(QObject* parent) : GroupAction(parent, "RecolorAction", true), _colorMapAction(this, "Color map", "example_c")
    {
        setText("Embedding Color Map");
        setObjectName("Embedding Color Map");
        _colorMapAction.setToolTip("Color map for recoloring ROI embedding based on top level embedding");
    }

    QMenu* getContextMenu(QWidget* parent = nullptr) override;

public:
    ColorMap2DAction& getColorMapAction() { return _colorMapAction; }

protected:
    ColorMap2DAction          _colorMapAction;        /** Color map action for top HSNE level */
};
