#include "ViewportSequence.h"

#include <QHeaderView>
#include <QFileDialog>
#include <QRectF>

#include <nlohmann/json.hpp>

#include <cmath>
#include <fstream>
#include <string>

/// //////////////////////// ///
/// ViewportSequence::Widget ///
/// //////////////////////// ///

ViewportSequence::Widget::Widget(QWidget* parent, ViewportSequence* viewportSequenceAction, const std::int32_t& widgetFlags) :
    WidgetActionWidget(parent, viewportSequenceAction),
    _tableView(this),
    _infoText(this)
{
    setMinimumHeight(200);

    /// Incoming connections ///
    connect(viewportSequenceAction, &ViewportSequence::dataModelChanged, this, &ViewportSequence::Widget::updateTableViewModel, Qt::DirectConnection);
    connect(viewportSequenceAction, &ViewportSequence::highlightRow, this, &ViewportSequence::Widget::updateTableHighlight, Qt::DirectConnection);

    /// Outgoing connections ///
    connect(&_tableView, &QTableView::clicked, this, [viewportSequenceAction](const QModelIndex& index) { viewportSequenceAction->setCurrentStepNum(index.row()); });

    auto layout = new QVBoxLayout();

    // TABLE VIEW
    _tableView.setSortingEnabled(false);
    _tableView.setStyleSheet("QTableView::indicator:checked{ padding: 10px; margin: 10px;}");

    _tableView.setSelectionBehavior(QAbstractItemView::SelectRows);

    auto horizontalHeader = _tableView.horizontalHeader();

    horizontalHeader->setStretchLastSection(false);
    horizontalHeader->setDefaultAlignment(Qt::AlignLeft);
    horizontalHeader->setSortIndicator(-1, Qt::DescendingOrder);
    horizontalHeader->setSectionResizeMode(QHeaderView::Stretch);

    _tableView.verticalHeader()->hide();
    _tableView.verticalHeader()->setDefaultSectionSize(5);

    layout->addWidget(&_tableView);

    // INFO TEXT
    _infoText.setReadOnly(true);
    _infoText.setFrameStyle(QFrame::Panel | QFrame::Plain);
    _infoText.setEnabled(false);
    _infoText.setLineWidth(0);
    _infoText.setFontPointSize(8);
    _infoText.setFixedHeight(18);
    _infoText.setAlignment(Qt::AlignCenter);
    _infoText.setText("Hover for limitations.");
    _infoText.setToolTip("When loading a sequence, the ImageViewer should have same extends as during saving. Also, clicking on the first entry yields unwanted results.");

    layout->addWidget(&_infoText);

    // BUTTONS UNDER TABLE
    auto toolbarLayout = new QHBoxLayout();
    toolbarLayout->setContentsMargins(0, 0, 0, 5);  // int left, int top, int right, int bottom

    toolbarLayout->addWidget(viewportSequenceAction->getStepBackAction().createWidget(this));
    toolbarLayout->addWidget(viewportSequenceAction->getStepForwardAction().createWidget(this));
    toolbarLayout->addWidget(viewportSequenceAction->getSeqLoadAction().createWidget(this));
    toolbarLayout->addWidget(viewportSequenceAction->getSeqSaveAction().createWidget(this));

    layout->addLayout(toolbarLayout);

    layout->addWidget(viewportSequenceAction->getViewportSharingActions().createWidget(this));

    layout->setContentsMargins(0, 0, 0, 0);
    setLayout(layout);

    updateTableViewModel(&viewportSequenceAction->getModel());

    // if data has first row, hightlight it
    if(_tableView.model()->rowCount() == 1)
        _tableView.selectRow(0);

}

void ViewportSequence::Widget::updateTableViewModel(QAbstractItemModel* model)
{
    _tableView.setModel(model);

    if (model->rowCount() == 0 || model->columnCount() == 0)
        return;
}

void ViewportSequence::Widget::updateTableHighlight(int row)
{
    _tableView.selectRow(row);
}

/// ////////////////////////// ///
/// ViewportSequence Utilities ///
/// ////////////////////////// ///

static bool readRoiSeq(const QString& fileName, ROIModel& roiModel)
{
    if (fileName.isEmpty())
        return false;

    std::ifstream loadFile(fileName.toStdString(), std::ios::in | std::ios::binary);

    if (!loadFile.is_open()) return false;

    // read a JSON file
    nlohmann::json viewports;

    try {
        loadFile >> viewports;
    }
    catch (nlohmann::detail::parse_error err)
    {
        Log::error("ViewportSequence::readRoiSeq: json parse error: " + std::string(err.what()));
        return false;
    }

    auto extractROI = [](nlohmann::json& element) -> utils::ROI {
        return { static_cast<uint32_t>(element.at(0).get<float>()), static_cast<uint32_t>(element.at(1).get<float>()), 
                 static_cast<uint32_t>(element.at(2).get<float>()), static_cast<uint32_t>(element.at(3).get<float>()),    // layer ROI
                 element.at(4).get<float>(), element.at(5).get<float>(), element.at(6).get<float>(), element.at(7).get<float>() };  // view ROI
    };

    // iterate the array
    for (auto& element : viewports) {
        roiModel.append(extractROI(element));
    }

    Log::info("ViewportSequence::readRoiSeq: Read viewport sequence from: " + fileName.toStdString());

    return true;
}

static bool writeRoiSeq(const QString& fileName, const ROIModel& roiModel)
{
    if (fileName.isEmpty())
        return false;

    if (roiModel.rowCount() == 0)
        return false;

    std::ofstream saveFile(fileName.toStdString(), std::ios::out | std::ios::trunc);

    if (!saveFile.is_open())
    {
        Log::error("ViewportSequence::saveSeq: Save file could not be opened.");
        return false;
    }

    // parse sequence view to json
    nlohmann::json viewports;

    size_t maxNumDigits = std::ceil(std::log10(roiModel.rowCount() + 1));

    for (int i = 0; i < roiModel.rowCount(); i++)
    {
        const auto roi = roiModel.dataRow(i);

        std::string stepName = std::to_string(i);
        if (stepName.length() < maxNumDigits)
            stepName.insert(stepName.front() == '-' ? 1 : 0, maxNumDigits - stepName.length(), '0');

        viewports[stepName] = {roi.layerBottomLeft.x(), roi.layerBottomLeft.y(), roi.layerTopRight.x(), roi.layerTopRight.y(),  // layer ROI
                               roi.viewRoiXY.x(),  roi.viewRoiXY.y(),  roi.viewRoiWH.x(),  roi.viewRoiWH.y() }; // view ROI
    }

    Log::info("ViewportSequence::writeRoiSeq: Save viewport sequence to: " + fileName.toStdString());

    // Write to file
    saveFile << std::setw(4) << viewports << std::endl;
    saveFile.close();

    return true;
}

/// //////////////// ///
/// ViewportSequence ///
/// //////////////// ///

ViewportSequence::ViewportSequence(QObject* parent) :
    GroupAction(parent, "ViewportSequence", true),
    _layout(),
    _stepBackAction(this, "Back"),
    _stepForwardAction(this, "Forward"),
    _seqLoadAction(this, "Load"),
    _seqSaveAction(this, "Save"),
    _currentStep(-1),
    _lockAddRoi(false),
    _dataModel(),
    _viewportSharingAction(this)
{
    setText("Viewports");
    setObjectName("Viewports");

    _stepBackAction.setToolTip("One step backward in sequence");
    _stepForwardAction.setToolTip("One step forwards in sequence");
    _seqLoadAction.setToolTip("Load viewport sequence from file");
    _seqSaveAction.setToolTip("Save viewport sequence to file");

    // STEP FORWARD AND BACKWARD IN SEQUENCE VIEW
    connect(&_stepBackAction, &TriggerAction::triggered, this, &ViewportSequence::stepBack);
    connect(&_stepForwardAction, &TriggerAction::triggered, this, &ViewportSequence::stepForward);

    // LOAD AND SAVE SEQUENCES
    const auto selectionFileFilter = tr("Text files (*.txt);;All files (*.*)");

    // Load dimension selection from file when the corresponding action is triggered
    connect(&_seqLoadAction, &TriggerAction::triggered, this, [this, selectionFileFilter]() {

        // prevent calling fileDialog.exec() or using a static method like QFileDialog::getOpenFileName
        // since they trigger some assertion failures due to threading issues
        QFileDialog* fileDialog = new QFileDialog(nullptr, tr("Load viewport sequence from file"), {}, selectionFileFilter);
        fileDialog->setAcceptMode(QFileDialog::AcceptOpen);
        fileDialog->setFileMode(QFileDialog::ExistingFile);

        connect(fileDialog, &QFileDialog::accepted, this, [this, fileDialog]() -> void {
            ROIModel roiModel;
            QString fileName = fileDialog->selectedFiles().first();

            if (readRoiSeq(fileName, roiModel))
                loadSeq(roiModel);
            else
                Log::warn("ViewportSequence: Could not read viewport sequence from" + fileName.toStdString());

            fileDialog->deleteLater();
            });

        fileDialog->open();
    });

    // Save viewport sequence to file when the corresponding action is triggered
    connect(&_seqSaveAction, &TriggerAction::triggered, this, [this, selectionFileFilter]() {

        QFileDialog* fileDialog = new QFileDialog(nullptr, tr("Write viewport sequence to file"), {}, selectionFileFilter);
        fileDialog->setAcceptMode(QFileDialog::AcceptSave);
        fileDialog->setFileMode(QFileDialog::AnyFile);
        
        connect(fileDialog, &QFileDialog::accepted, this, [this, fileDialog]() -> void {
            ROIModel roiModel;
            QString fileName = fileDialog->selectedFiles().first();;
            writeRoiSeq(fileName, _dataModel);
            fileDialog->deleteLater();
        });

        fileDialog->open();
    });

    connect(&_viewportSharingAction, &ViewportSharingActions::viewportChanged, this, [this](const QVector3D layerRoiBottomLeft, const QVector3D layerRoiTopRight, const QVector3D viewRoiXY, const QVector3D viewRoiWH) {
        if (_dataModel.rowCount() != 0)
            return;

        utils::ROI roi;
        roi.layerBottomLeft = { layerRoiBottomLeft.x(), layerRoiBottomLeft.y() };
        roi.layerTopRight = { layerRoiTopRight.x(), layerRoiTopRight.y() };
        roi.viewRoiXY = { viewRoiXY.x(), viewRoiXY.y() };
        roi.viewRoiWH = { viewRoiWH.x(), viewRoiWH.y() };

        appendROI(roi);
        });

}

void ViewportSequence::setCurrentStepNum(int step)
{
    if (step >= _dataModel.rowCount() || step < 0)
    {
        Log::warn("ViewportSequence::setCurrentStepNum: step outside bounds");
        return;
    }

    _lockAddRoi = true;

    _currentStep = step;
    emit highlightRow(_currentStep);
    triggerViewportChange(_dataModel.dataRow(_currentStep));
}

void ViewportSequence::appendROI(const utils::ROI& roi)
{
    // check if the roi needs to be appended or whether we are stepping through the sequence history
    if (_lockAddRoi)
    {
        _lockAddRoi = false;
        return;
    }

    addROIToModel(roi);
}

void ViewportSequence::addROIToModel(const utils::ROI& roi)
{
    // For selection ROI update, set the default view ROI
    if (roi.viewRoiWH.x() == 0.f && roi.viewRoiWH.y() == 0.f) {
        utils::ROI roiCopy = roi;

        auto firstROI = _dataModel.dataRow(0);

        roiCopy.viewRoiXY = { firstROI.viewRoiXY.x(), firstROI.viewRoiXY.y() };
        roiCopy.viewRoiWH = { firstROI.viewRoiWH.x(), firstROI.viewRoiWH.y() };
        _dataModel.append(roiCopy);
    }
    else {
        _dataModel.append(roi);
    }

    emit dataModelChanged(&_dataModel);

    _currentStep = _dataModel.rowCount() - 1;
    emit highlightRow(_currentStep);
}

void ViewportSequence::triggerViewportChange(const utils::ROI& roi)
{
    QRectF viewRectangle;
    viewRectangle.setX(roi.viewRoiXY.x());
    viewRectangle.setY(roi.viewRoiXY.y());
    viewRectangle.setWidth(roi.viewRoiWH.x());
    viewRectangle.setHeight(roi.viewRoiWH.y());

    Log::warn(fmt::format("ViewportSequence::viewRectangle {} {} {} {}", viewRectangle.left(), viewRectangle.right(), viewRectangle.top(), viewRectangle.bottom()));

    _viewportSharingAction.setViewROI(viewRectangle.left(), viewRectangle.right(), viewRectangle.top(), viewRectangle.bottom());

    emit updatedROIInSequenceView(roi);
}

void ViewportSequence::stepBack() 
{
    if (_currentStep <= 0)
        return;
    
    _lockAddRoi = true;

    _currentStep--;
    
    emit highlightRow(_currentStep);
    triggerViewportChange(_dataModel.dataRow(_currentStep));
}

void ViewportSequence::stepForward() 
{
    if (_currentStep >= _dataModel.rowCount() - 1)
        return;

    _lockAddRoi = true;

    _currentStep++;

    emit highlightRow(_currentStep);
    triggerViewportChange(_dataModel.dataRow(_currentStep));
}

void ViewportSequence::setROIModel(const ROIModel& roiModel)
{
    _dataModel.reset();

    for (int i = 0; i < roiModel.rowCount(); i++)
    {
        _dataModel.append(roiModel.dataRow(i));
    }

    _currentStep = 0;
    emit highlightRow(_currentStep);
}


bool ViewportSequence::loadSeq(const ROIModel& roiModel)
{
    setROIModel(roiModel);

    if (roiModel.rowCount() > 0)
        triggerViewportChange(roiModel.dataRow(0));

    Log::info("ViewportSequence::loadSeq: Loaded viewport sequence");

    return true;
}

bool ViewportSequence::saveSeq(const QString& fileName) const
{
    return writeRoiSeq(fileName, _dataModel);
}



