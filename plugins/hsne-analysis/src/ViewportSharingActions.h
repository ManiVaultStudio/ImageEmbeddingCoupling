#pragma once

#include "actions/GroupAction.h"
#include <actions/IntegralRectangleAction.h>
#include <actions/DecimalRectangleAction.h>

#include <QVector3D>
#include <QRect>
#include <QRectF>

using namespace mv::gui;

/** Share zoom coordinates between image viewers and analysis plugin
  layerROI -> image coordinates
  viewROI  -> world coordinates(depends on viewer size)
*/
class ViewportSharingActions : public mv::gui::GroupAction
{
    Q_OBJECT;

public:

    /**
     * Constructor
     * @param parent Pointer to parent object
     */
    Q_INVOKABLE ViewportSharingActions(QObject* parent);

signals:
    // send from image viewer when the viewport changed
    void viewportChanged(const QVector3D layerRoiBottomLeft, const QVector3D layerRoiTopRight, const QVector3D viewRoiXY, const QVector3D viewRoiWH);

public: // Action getters

    IntegralRectangleAction& getRoiLayerAction() { return _roiLayerAction; }
    DecimalRectangleAction& getRoiViewAction() { return _roiViewAction; }

public: // setter

    void setViewROI(float left, float right, float bottom, float top);

protected:
    IntegralRectangleAction     _roiSelectionAction;/** Selection region of interest action */
    IntegralRectangleAction     _roiLayerAction;    /** Layer region of interest action */
    DecimalRectangleAction      _roiViewAction;     /** View region of interest action */

private:
    QRect                       _selectionROI;
    QRect                       _layerROI;
    QRectF                      _viewROI;
};
