#include "ViewportSharingActions.h"

#include "Logger.h"

ViewportSharingActions::ViewportSharingActions(QObject* parent) :
    GroupAction(parent, "HsneImageViewportSharingAction", true),
    _roiSelectionAction(this, "Selection ROI"),
    _roiLayerAction(this, "Layer ROI"),
    _roiViewAction(this, "View ROI"),
    _selectionROI(),
    _layerROI(),
    _viewROI()
{
    setText("HSNE Image Selection");
    
    _roiSelectionAction.getRangeAction(IntegralRectangleAction::Axis::X).getRangeMinAction().setPrefix("");
    _roiSelectionAction.getRangeAction(IntegralRectangleAction::Axis::X).getRangeMaxAction().setPrefix("");
    _roiSelectionAction.getRangeAction(IntegralRectangleAction::Axis::Y).getRangeMinAction().setPrefix("");
    _roiSelectionAction.getRangeAction(IntegralRectangleAction::Axis::Y).getRangeMaxAction().setPrefix("");

    _roiLayerAction.getRangeAction(IntegralRectangleAction::Axis::X).getRangeMinAction().setPrefix("");
    _roiLayerAction.getRangeAction(IntegralRectangleAction::Axis::X).getRangeMaxAction().setPrefix("");
    _roiLayerAction.getRangeAction(IntegralRectangleAction::Axis::Y).getRangeMinAction().setPrefix("");
    _roiLayerAction.getRangeAction(IntegralRectangleAction::Axis::Y).getRangeMaxAction().setPrefix("");

    _roiViewAction.getRangeAction(DecimalRectangleAction::Axis::X).getRangeMinAction().setPrefix("");
    _roiViewAction.getRangeAction(DecimalRectangleAction::Axis::X).getRangeMaxAction().setPrefix("");
    _roiViewAction.getRangeAction(DecimalRectangleAction::Axis::Y).getRangeMinAction().setPrefix("");
    _roiViewAction.getRangeAction(DecimalRectangleAction::Axis::Y).getRangeMaxAction().setPrefix("");

    _roiSelectionAction.setToolTip("Selection IDs, manually updated in Image Viewer (bottom-left:x, bottom-left:y, top-right:x, top-right:y)");
    _roiLayerAction.setToolTip("Layer region of interest discrete image coordinates (bottom-left:x, bottom-left:y, top-right:x, top-right:y)");
    _roiViewAction.setToolTip("View region of interest in fractional world coordinates (bottom-left:x, bottom-left:y, top-right:x, top-right:y)");

    addAction(&_roiLayerAction);
    addAction(&_roiViewAction);   
    addAction(&_roiSelectionAction);

    _roiLayerAction.setConnectionPermissionsFlag(ConnectionPermissionFlag::All, false, true);
    _roiViewAction.setConnectionPermissionsFlag(ConnectionPermissionFlag::All, false, true);
    _roiSelectionAction.setConnectionPermissionsFlag(ConnectionPermissionFlag::All, false, true);
    
    connect(&_roiViewAction, &DecimalRectangleAction::rectangleChanged, this, [this](float left, float right, float bottom, float top) {
        QRect layerROI;
        layerROI.setLeft(_roiLayerAction.getLeft());
        layerROI.setRight(_roiLayerAction.getRight());
        layerROI.setBottom(_roiLayerAction.getBottom());
        layerROI.setTop(_roiLayerAction.getTop());

        QRectF viewROI = QRectF{ left, bottom, right - left, top - bottom };

        if (layerROI != _layerROI) {
            _layerROI = layerROI;
            _viewROI = viewROI;

            // The image viewer internally it uses a flipped y axis. Here we use the coordinates system as the user would expect it
            const auto layerRoiBottomLeft = QVector3D(_layerROI.left(), _layerROI.bottom(), 0.f);
            const auto layerRoiTopRight = QVector3D(_layerROI.right(), _layerROI.top(), 0.f);

            const auto viewRoiXY = QVector3D(_viewROI.x(), _viewROI.y(), 0.f);
            const auto viewRoiWH = QVector3D(_viewROI.width(), _viewROI.height(), 0.f);

            emit viewportChanged(layerRoiBottomLeft, layerRoiTopRight, viewRoiXY, viewRoiWH);
        }
        });

    connect(&_roiSelectionAction, &IntegralRectangleAction::rectangleChanged, this, [this](int32_t left, int32_t right, int32_t bottom, int32_t top) {
        QRect selectionROI;
        selectionROI.setLeft(left);
        selectionROI.setRight(right);
        selectionROI.setBottom(bottom);
        selectionROI.setTop(top);

        if (selectionROI != _selectionROI) {
            _selectionROI = selectionROI;

            // The image viewer internally it uses a flipped y axis. Here we use the coordinates system as the user would expect it
            const auto layerRoiBottomLeft = QVector3D(_selectionROI.left(), _selectionROI.bottom(), 0.f);
            const auto layerRoiTopRight = QVector3D(_selectionROI.right(), _selectionROI.top(), 0.f);

            const auto viewRoiXY = QVector3D(-1.f, -1.f, 0.f);
            const auto viewRoiWH = QVector3D(0.f, 0.f, 0.f);

            emit viewportChanged(layerRoiBottomLeft, layerRoiTopRight, viewRoiXY, viewRoiWH);
        }
        });

}

void ViewportSharingActions::setViewROI(float left, float right, float bottom, float top) {
    _roiViewAction.setRectangle(left, right, bottom, top);
}
