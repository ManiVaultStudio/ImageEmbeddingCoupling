#include "DimensionSelectionAction.h"

DimensionSelectionAction::DimensionSelectionAction(QObject* parent) :
    GroupAction(parent, "DimensionSelectionAction"),
    _pickerAction(this, "DimensionsPickerAction")
{
    setText("Dimensions");
}