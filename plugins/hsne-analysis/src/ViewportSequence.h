#pragma once

#include "ViewportSharingActions.h"
#include "Utils.h"

#include "actions/GroupAction.h"
#include "actions/TriggerAction.h"

#include <QAbstractTableModel>
#include <QList>
#include <QTableView>
#include <QVBoxLayout>
#include <QTextEdit>

using namespace mv::gui;

/**
 * Data model for table viewer
 * 
 * @author Alexander Vieth
 */
class ROIModel : public QAbstractTableModel {
    QList<utils::ROI> m_data;
public:
    ROIModel(QObject* parent = {}) : QAbstractTableModel{ parent } { }

    auto rowCount() const { return m_data.count(); }
    int rowCount(const QModelIndex&) const override { return rowCount(); }
    constexpr int columnCount() const { return 5; }
    int columnCount(const QModelIndex&) const override { return columnCount(); }

    /** Returns a specific row from the table data */
    utils::ROI dataRow(const int row) const {
        return m_data[row];
    }

    /** Defines which data in shown in each column*/
    QVariant data(const QModelIndex& index, int role) const override {
        if (role != Qt::DisplayRole && role != Qt::EditRole) return {};
        const auto& roi = m_data[index.row()];
        switch (index.column()) {
        case 0: return index.row();
        case 1: return roi.layerBottomLeft.x();
        case 2: return roi.layerBottomLeft.y();
        case 3: return roi.layerTopRight.x();
        case 4: return roi.layerTopRight.y();
        default: return {};
        };
    }

    /** Sets column label in table header */
    QVariant headerData(int section, Qt::Orientation orientation, int role) const override {
        if (orientation != Qt::Horizontal || role != Qt::DisplayRole) return {};
        switch (section) {
        case 0: return "ID";
        case 1: return "layerBottomLeft.x";
        case 2: return "layerBottomLeft.y";
        case 3: return "layerTopRight.x";
        case 4: return "layerTopRight.y";
        default: return {};
        }
    }

    /** Appends one row to the table model */
    void append(const utils::ROI& roi) {
        beginInsertRows({}, m_data.count(), m_data.count());
        m_data.append(roi);
        endInsertRows();
    }
    
    /** Removes all data from the model */
    void reset()
    {
        if (m_data.isEmpty())
            return;

        beginRemoveRows({}, 0, rowCount() - 1);
        m_data.clear();
        endRemoveRows();
    }

};

/** Describes the navigation direction in the sequence history */
enum class SequenceDirection : int32_t {
    FORWARD = 1,
    BACKWARD = -1
};

/**
 * Shows sequence of image viewer viewports
 * 
 * @author Alexander Vieth
 */
class ViewportSequence : public GroupAction
{
    Q_OBJECT

protected:
    /** Widget class for dimension selection action */
    class Widget : public WidgetActionWidget {
    public:

        /**
        * Constructor
        * @param parent Pointer to parent widget
        * @param dimensionSelectionAction Pointer to dimension selection action
        * @param widgetFlags Widget flags for the configuration of the widget
        */
        Widget(QWidget* parent, ViewportSequence* viewportSequenceAction, const std::int32_t& widgetFlags);

    protected:

        /**
        * Update the table view source model
        * @param model Pointer to table view source model
        */
        void updateTableViewModel(QAbstractItemModel* model);

        /** Highlights a row in the table */
        void updateTableHighlight(int row);

    protected:
        QTableView      _tableView;     // actual ui table
        QTextEdit       _infoText;      // Info for the user, shown above interaction buttons
    };

    /**
    * Get widget representation of the dimension selection action
    * @param parent Pointer to parent widget
    * @param widgetFlags Widget flags for the configuration of the widget
    */
    QWidget* getWidget(QWidget* parent, const std::int32_t& widgetFlags) override {
        return new Widget(parent, this, widgetFlags);
    };

public:

    /**
     * Constructor
     * @param parent Pointer to parent object
     */
    ViewportSequence(QObject* parent);

    /** Adds a data row to the table's end */
    void appendROI(const utils::ROI& roi);

    /** Returns current data model number */
    utils::ROI getCurrentROI(int row) const { return _dataModel.dataRow(row); };

    /** Returns current table row number */
    int getCurrentStepNum() const { return _currentStep; };

    void setCurrentStepNum(int step);
    
    bool getLockedAddRoi() const { return _lockAddRoi; };

    void setLockedAddRoi(bool state) { _lockAddRoi = state; };

    /** Go one step backwards in sequence history */
    void stepBack();

    /** Go one step forward in sequence history */
    void stepForward();

    /** Reset data in table and use the new given data */
    void setROIModel(const ROIModel& roiModel);

public: // UI getters

    ROIModel& getModel() { return _dataModel; };
    TriggerAction& getStepBackAction() { return _stepBackAction; }
    TriggerAction& getStepForwardAction() { return _stepForwardAction; }
    TriggerAction& getSeqLoadAction() { return _seqLoadAction; }
    TriggerAction& getSeqSaveAction() { return _seqSaveAction; }
    ViewportSharingActions& getViewportSharingActions() { return _viewportSharingAction; }

private:

    /** Load viewport sequence from file, triggered through UI */
    bool loadSeq(const ROIModel& roiModel);

    /** Save viewport sequence to file, triggered through UI */
    bool saveSeq(const QString& fileName) const;

    /** Add a roi to the data model and update the highlight */
    void addROIToModel(const utils::ROI& roi);

    // sets the viewport in the image viewer to roi and triggers an hsne update
    void triggerViewportChange(const utils::ROI& roi);

signals:

    /**
     * Signals that the proxy model changed
     * @param dimensionsPickerProxyModel Pointer to dimensions picker proxy model
     */
    void dataModelChanged(ROIModel* roiModel);

    void highlightRow(int row);

    void updatedROIInSequenceView(const utils::ROI& roi);

protected:
    QVBoxLayout     _layout;
    ROIModel        _dataModel;

    ViewportSharingActions  _viewportSharingAction;        /** Viewport sharing action */

    TriggerAction   _stepBackAction;
    TriggerAction   _stepForwardAction;
    TriggerAction   _seqLoadAction;
    TriggerAction   _seqSaveAction;

    int             _currentStep;       // current table row
    bool            _lockAddRoi;        // don't add roi while going back- and forward
};