#include "HsneScaleAction.h"
#include "HsneHierarchy.h"
#include "TsneSettingsAction.h"
#include "DataHierarchyItem.h"
#include "InteractiveHsnePlugin.h"
#include "UtilsScale.h"
#include "Utils.h"
#include "RegularHsneAction.h"
#include "PCA.h"

#include <QMenu>
#include <QSize>

#include <algorithm>
#include <numeric>
#include <array>
#include <iterator>

/// ////////////////// ///
/// SETTING CONVERSION ///
/// ////////////////// ///

math::PCA_ALG convertPcaAlgorithm(size_t index) {
    math::PCA_ALG alg = math::PCA_ALG::COV;

    switch (index)
    {
    case 0:
        alg = math::PCA_ALG::SVD;
        break;
    case 1:
        alg = math::PCA_ALG::COV;
        break;
    }

    return alg;
}


/// /////////////// ///
/// HsneScaleAction ///
/// /////////////// ///

HsneScaleAction::HsneScaleAction(QObject* parent, InteractiveHsnePlugin* hsneAnalysisPlugin, TsneSettingsAction& tsneSettingsAction, HsneHierarchy& hsneHierarchy,
    Dataset<Points> inputDataset, Dataset<Points> embeddingDataset, Dataset<Points> firstEmbedding, Dataset<Points> topLevelLandmarkData, 
    Dataset<Points> pointInitTypesDataset, Dataset<Points> roiRepresentationDataset, Dataset<Points> numberTransitions, Dataset<Points> colorScatterRoiHSNE,
    Dataset<Points> regHsneTopLevel) :
    GroupAction(parent, "HsneScaleAction", true),
    _hsneAnalysisPlugin(hsneAnalysisPlugin),
    _input(inputDataset),
    _tsneSettingsAction(tsneSettingsAction),
    _hsneHierarchy(hsneHierarchy),
    _embedding(embeddingDataset),
    _firstEmbedding(firstEmbedding),
    _regHsneTopLevel(regHsneTopLevel),
    _topLevelLandmarkData(topLevelLandmarkData),
    _pointInitTypes(pointInitTypesDataset),
    _roiRepresentation(roiRepresentationDataset),
    _numberTransitions(numberTransitions),
    _colorScatterRoiHSNE(colorScatterRoiHSNE),
    _currentScaleLevel(0),
    _numImagePoints(0),
    _inputImageSize(),
    _updateStopAction(this, "Stop updating"),
    _fixScaleAction(this, "Fix scale level"),
    _scaleUpDownActions(this),
    _thresholdAction(this, "Influence tresh"),
    _influenceHeuristic(this, "Influence heuristic", true),
    _visRangeAction(this, "Vis Range"),
    _visBudgetMinAction(this, "Min Vis Budget"),
    _visBudgetMaxAction(this, "Max Vis Budget"),
    _visBudgetTargetAction(this, "Vis budget target"),
    _rangeHeuristicAction(this, "Budget heuristic"),
    _landmarkFilterSlider(this, "Min transitions"),
    _landmarkFilterToggle(this, "Filter Landmarks"),
    _colorMapRoiEmbAction(this, "Color map", "example_c"),
    _colorMapFirstEmbAction(new RecolorAction(this)),
    _recolorDuringUpdates(this, "Recolor during gradient descent", true),
    _recomputeScale(this, "Recompute scale embedding"),
    _embScalingSlider(this, "Scaling multiplier"),
    _noExaggerationUpdate(this, "No exaggeration for new embeddings", false),
    _randomInitMeta(this, "Update init meta data", false),
    _compRepresents(this, "Compute representations"),
    _copySelectedAttributes(this, "Selection to Dataset"),
    _visualRange(0u), // 2'000u
    _visBudgetMax(50'000u),
    _lockBudgetSlider(false), 
    _currentScaleAction(this, "Current scale", StatusAction::Status::Info, "not initialized"),
    _embScaleFac(this, "Scaling factor", StatusAction::Status::Info, "not initialized"),
    _embCurrExt(this, "Current extends", StatusAction::Status::Info, "not initialized"),
    _embMaxExt(this, "Reference extends", StatusAction::Status::Info, "not initialized"),
    _tresh_influence(-1.0f),
    _currentEmbExtends(-1, -1, -1, -1),
    _refEmbExtends(-1, -1, -1, -1),
    _hsneScaleUpdate(hsneHierarchy),
    _updateRoiImageLock(10),
    _tsneAnalysis("HSNE"),
    _RoiGoodForUpdate(true),
    _updateMetaDataset(false)
{
    /// UI set up: global values
    setText("HSNE scale");
    setLabelSizingType(LabelSizingType::Fixed);
    setLabelWidthFixed(100);

    /// UI set up: add actions
    for (auto& action : WidgetActions { &_updateStopAction, &_thresholdAction, &_influenceHeuristic, &_visRangeAction,
        &_visBudgetMinAction, &_visBudgetMaxAction, &_visBudgetTargetAction, &_rangeHeuristicAction, &_currentScaleAction,
        & _scaleUpDownActions,& _fixScaleAction,& _landmarkFilterSlider,& _landmarkFilterToggle,& _colorMapRoiEmbAction,
        _colorMapFirstEmbAction,& _recolorDuringUpdates,& _embScalingSlider,& _embScaleFac,& _embCurrExt,& _embMaxExt,
        & _noExaggerationUpdate,& _recomputeScale,& _randomInitMeta,& _compRepresents,& _copySelectedAttributes })
        addAction(action);

    /// UI set up: _updateStopAction
    _updateStopAction.setToolTip("Stop refining the selected landmarks on panning and zooming");
    _updateStopAction.setChecked(false);

    /// UI set up: _fixScaleAction
    _fixScaleAction.setToolTip("Fixes the current scale level");
    _fixScaleAction.setChecked(false);

    /// UI set up: _noExaggerationUpdate
    _noExaggerationUpdate.setToolTip("Use no exaggeration for each new embedding.");

    /// UI set up: landmark influence heuristic and thresholding
    {
        // INFO: Currently not used
        // The selection mapping is based on the heuristical approach (influenceHierarchy)
        // It is not useful to mix both thresholded and heuristical approaches
        // HsneHierarchy::computeSelectionMapsAtScale would need to use HsneHierarchy::getLocalIDsInRefinedScale with _tresh_influence

        _thresholdAction.initialize(0.0f, 1.0f, 0.1f, 2u);
        _thresholdAction.setSingleStep(0.01f);
        _thresholdAction.setToolTip("Set landmark influence treshold");
        _thresholdAction.setDefaultWidgetFlags(IntegralAction::SpinBox | IntegralAction::Slider);
        _thresholdAction.setEnabled(false);
        _thresholdAction.setVisible(false);

        _influenceHeuristic.setToolTip("Use heuristic to determine landmark representation");
        _influenceHeuristic.setChecked(true);
        _influenceHeuristic.setVisible(false);

        connect(&_thresholdAction, &DecimalAction::valueChanged, this, [this](const float& val) {
            _tresh_influence = val;
            });

        connect(&_influenceHeuristic, &ToggleAction::toggled, this, [this](const bool& val) {
            _thresholdAction.setEnabled(!_influenceHeuristic.isChecked());

            if(_influenceHeuristic.isChecked())
                _tresh_influence = -1.0f;
            else
                _tresh_influence = _thresholdAction.getValue();
            });
    }

    /// UI set up: visual budget range ///
    {
        // Set default values for visual budget
        uint32_t minVisBudgetStart = 4'000u;
        uint32_t maxVisBudgetStart = minVisBudgetStart + _visualRange;

        //_rangeHeuristicAction.setToolTip("Stop scale traversal once the number of landmarks falls within the budget range");
        _rangeHeuristicAction.setToolTip("Check if it makes more sense to go down instead of up");
        _rangeHeuristicAction.setChecked(true);
        _rangeHeuristicAction.setVisible(true);

        _visRangeAction.setDefaultWidgetFlags(IntegralAction::SpinBox | IntegralAction::Slider);
        _visRangeAction.initialize(1, _visBudgetMax, _visualRange);
        _visRangeAction.setVisible(false);

        _visBudgetMinAction.setDefaultWidgetFlags(IntegralAction::SpinBox | IntegralAction::Slider);
        _visBudgetMinAction.initialize(1, _visBudgetMax, minVisBudgetStart);
        _visBudgetMinAction.setVisible(false);

        _visBudgetMaxAction.setDefaultWidgetFlags(IntegralAction::SpinBox | IntegralAction::Slider);
        _visBudgetMaxAction.initialize(1, _visBudgetMax, maxVisBudgetStart);
        _visBudgetMaxAction.setVisible(false);

        _visBudgetTargetAction.setDefaultWidgetFlags(IntegralAction::SpinBox | IntegralAction::Slider);
        _visBudgetTargetAction.initialize(1, 100'000, 10'000);
        _visBudgetTargetAction.setToolTip("Scale with closest number of representative landmarks is selected");
    }

    /// UI set up: treshold for filtering landmarks ///
    {
        _landmarkFilterSlider.setToolTip("Minimum number of landmarks connections on scale for given selection");
        _landmarkFilterToggle.setToolTip("Check for minimum number of connections");

        _landmarkFilterSlider.setDefaultWidgetFlags(IntegralAction::SpinBox | IntegralAction::Slider);
        _landmarkFilterSlider.initialize(1, 100, 25);
        _landmarkFilterSlider.setDisabled(true);
    }

    /// UI set up: recompute trigger ///
    _recomputeScale.setToolTip("Recompute scale ROI HSNE embedding with random init \n(Does not take in account new settings like Filtering)");
    _randomInitMeta.setToolTip("Update init meta data on random init recompute.");
    _randomInitMeta.setVisible(false);
    connect(&_recomputeScale, &TriggerAction::triggered, this, [this]() {
        recomputeScaleEmbedding(/*_randomInitMeta.isChecked()*/);
        });

    /// UI set up: compute representations trigger ///
    _compRepresents.setToolTip("Go up the scales and find nice representations. UNFINISHED");
    connect(&_compRepresents, &TriggerAction::triggered, this, [this]() {
        compRepresents();
        });

    /// UI set up: publish data of selected data points ///
    _copySelectedAttributes.setToolTip("Copy the attributes of all currently selected data items into (and newly populate) the 'Selection Data' data set.");
    connect(&_copySelectedAttributes, &TriggerAction::triggered, this, [this]() {
        publishSelectionData();
        });

    /// UI set up: Color mapping ///
    {
        _colorMapRoiEmbAction.setToolTip("Image color map");

        // Disable horizontal range actions
        _colorMapRoiEmbAction.getRangeAction(ColorMapAction::Axis::X).getRangeMinAction().setEnabled(false);
        _colorMapRoiEmbAction.getRangeAction(ColorMapAction::Axis::X).getRangeMaxAction().setEnabled(false);

        // Disable vertical range actions
        _colorMapRoiEmbAction.getRangeAction(ColorMapAction::Axis::Y).getRangeMinAction().setEnabled(false);
        _colorMapRoiEmbAction.getRangeAction(ColorMapAction::Axis::Y).getRangeMaxAction().setEnabled(false);

        // Disable discrete action
        _colorMapRoiEmbAction.getDiscretizeAction().setEnabled(false);

        _recolorDuringUpdates.setToolTip("Toggles whether recoloring should happen only at the end of the gradient descent or continuously");

        // Add action to set color map for recoloring based on top level embedding
        _firstEmbedding->addAction(*_colorMapFirstEmbAction);
    }

    /// UI set up: Scaling factor for embedding during update ///
    {
        _embScalingSlider.setToolTip("Scaling factor for embedding during update.");
        _embScalingSlider.setDefaultWidgetFlags(DecimalAction::SpinBox | DecimalAction::Slider);
        _embScalingSlider.initialize(0.0001f, 10, 1, 4);

        connect(&_embScalingSlider, &DecimalAction::valueChanged, this, &HsneScaleAction::updateEmbScaling);

        _embScaleFac.setToolTip("In brackest the scaling factor without scaling divisor applied.");
        _embMaxExt.setToolTip("Embedding extends of top level embedding after 100 iterations");
    }

    /// Connect _tsneAnalysis ///
    {
        connect(&_tsneAnalysis, &TsneAnalysis::finished, this, [this]() {
            Log::info("HsneScaleAction::TsneAnalysis::finished");
            utils::ScopedTimer TsneAnalysisFinished("HsneScaleAction::TsneAnalysis::finished connection");

            // compute new embedding extends
            const auto& embContainer = _tsneAnalysis.getEmbedding();
            setCurrentEmbExtends(utils::computeExtends(embContainer.getData()));

            // update UI
            _hsneAnalysisPlugin->getHsneSettingsAction().getGeneralHsneSettingsAction().getInitAction().setEnabled(false);
            _hsneAnalysisPlugin->getHsneSettingsAction().setReadOnly(false);
            _hsneAnalysisPlugin->getHsneSettingsAction().getTsneSettingsAction().setReadOnly(false);

            // update recolored image and recoloring of current embedding based on top level emb
            _hsneAnalysisPlugin->setColorMapDataRoiHSNE();

            // Save first Top Scale embedding, recoloring and scatter colors
            if (_firstEmbedding->getProperty("Init").toBool() == false)
            {
                _firstEmbedding->setProperty("Init", true);

                // set top level embedding
                const auto& embContainer = _tsneAnalysis.getEmbedding();
                _firstEmbedding->setData(embContainer.getData().data(), embContainer.getNumPoints(), 2);
                events().notifyDatasetDataChanged(_firstEmbedding);

                // set top level regular HSNE embedding
                _regHsneTopLevel->setData(embContainer.getData().data(), embContainer.getNumPoints(), 2);
                events().notifyDatasetDataChanged(_regHsneTopLevel);

                // set scatter colors for regular HSNE embedding, use default color map
                {
                    std::vector<float> scatterColors;
                    const auto currentColormap = _colorMapRoiEmbAction.getColorMap();
                    _colorMapRoiEmbAction.setColorMap(currentColormap);
                    _hsneAnalysisPlugin->setScatterColorMapData(_regHsneTopLevel, _regTopLevelScatterCol, _colorMapRoiEmbAction.getColorMapImage(), scatterColors);
                }

                // save selection map
                _hsneAnalysisPlugin->getSelectionMapTopLevelEmbLocalToBottom() = _hsneAnalysisPlugin->getSelectionMapLocalToBottom();
                _hsneAnalysisPlugin->getSelectionMapTopLevelEmbBottomToLocal() = _hsneAnalysisPlugin->getSelectionMapBottomToLocal();

                // set color mapping of top level embedding
                _hsneAnalysisPlugin->setColorMapDataTopLevelEmb();

                //_hsneAnalysisPlugin->getHsneSettingsAction().getMeanShiftActionAction().compute();

                // compute new emb colors based on representative landmarks in top level embedding  
                _hsneAnalysisPlugin->setScatterColorBasedOnTopLevel();
            }

            // set gradient descent iterations for updates to lower number
            _hsneAnalysisPlugin->getHsneSettingsAction().getTsneSettingsAction().getGeneralTsneSettingsAction().getNumDefaultUpdateIterationsActionAction().setValue(500);

            emit finished();
            });

        connect(&_tsneAnalysis, &TsneAnalysis::publishExtends, this, &HsneScaleAction::setRefEmbExtends);

        // Update embedding points when the TSNE analysis produces new data
        connect(&_tsneAnalysis, &TsneAnalysis::embeddingUpdate, this, [this](const std::vector<float>& emb, const uint32_t& numPoints, const uint32_t& numDimensions) {

            // Update the refine embedding with new data
            _embedding->setData(emb, numDimensions);

            // Set updated interation count in UI
            _tsneSettingsAction.getGeneralTsneSettingsAction().getNumComputatedIterationsAction().setValue(_tsneAnalysis.getNumIterations() - 1);

            // Update the color map every 100 iterations
            if (_recolorDuringUpdates.isChecked() && !utils::CyclicLock::isLocked(++_updateRoiImageLock))
                _hsneAnalysisPlugin->setColorMapDataRoiHSNE();

            // Notify others that the embedding points have changed
            events().notifyDatasetDataChanged(_embedding);

            // Update meta data only once, at the first embeddingUpdate after _hsneScaleUpdate is finished in order to resize the datasets correctly
            // Meta data is not updated for the top level embedding, it is set earlier in computeTopLevelEmbedding()
            if (_updateMetaDataset)
            {
                assert(_hsneScaleUpdate.getInitTypes().size() == numPoints);
                _pointInitTypes->setData(_hsneScaleUpdate.getInitTypes().data(), numPoints, 1);
                events().notifyDatasetDataChanged(_pointInitTypes);

                assert(_hsneScaleUpdate.getRoiRepresentationFractions().size() == numPoints);
                _roiRepresentation->setData(_hsneScaleUpdate.getRoiRepresentationFractions().data(), numPoints, 1);
                events().notifyDatasetDataChanged(_roiRepresentation);

                assert(_hsneScaleUpdate.getNumberTransitions().size() == numPoints);
                _numberTransitions->setData(_hsneScaleUpdate.getNumberTransitions().data(), numPoints, 1);
                events().notifyDatasetDataChanged(_numberTransitions);

                std::vector<float> tempResize(numPoints * 3u, 0.0f);
                _colorScatterRoiHSNE->setData(tempResize.data(), numPoints, 3);
                events().notifyDatasetDataChanged(_colorScatterRoiHSNE);

                // save color image as prev
                _hsneAnalysisPlugin->saveCurrentColorImageAsPrev();

                // compute new color image
                _hsneAnalysisPlugin->setColorMapDataRoiHSNE();

                // compute new emb colors based on representative landmarks in top level embedding  
                _hsneAnalysisPlugin->setScatterColorBasedOnTopLevel();

                // set currentLevelLandmarkData and respective selection mappings. TODO: reduce code duplication
                {
                    std::vector<float> dataLandmarks;
                    std::vector<uint32_t> enabledDimensionsIDs;
                    size_t numEnabledDimensions;
                    std::vector<uint32_t> imageIDs;

                    // Set selection linking for landmark data
                    auto& mapCurrentLevelDataLocalToBottom = _hsneAnalysisPlugin->getSelectionMapCurrentLevelDataLocalToBottom();
                    auto& mapCurrentLevelDataBottomToLocal = _hsneAnalysisPlugin->getSelectionMapCurrentLevelDataBottomToLocal();
                    mapCurrentLevelDataLocalToBottom.clear();
                    mapCurrentLevelDataLocalToBottom.resize(_idMap.size());
                    mapCurrentLevelDataBottomToLocal.clear();
                    mapCurrentLevelDataBottomToLocal.resize(_input->getNumPoints());

                    // Get global landmark IDs
                    for (const auto& [dataID, embIdAndPos] : _idMap)
                    {
                        // add selection map entry
                        mapCurrentLevelDataLocalToBottom[embIdAndPos.posInEmbedding].emplace_back(dataID);
                        mapCurrentLevelDataBottomToLocal[dataID] = embIdAndPos.posInEmbedding;

                        // copy data ID
                        imageIDs.emplace_back(dataID);
                    }
                    std::sort(utils::exec_policy, imageIDs.begin(), imageIDs.end());

                    // Get dimensions
                    std::tie(enabledDimensionsIDs, numEnabledDimensions) = _hsneAnalysisPlugin->enabledDimensions();

                    // get data
                    dataLandmarks.resize(enabledDimensionsIDs.size()* imageIDs.size());
                    _input->populateDataForDimensions<std::vector<float>, std::vector<uint32_t>, std::vector<uint32_t>>(dataLandmarks, enabledDimensionsIDs, imageIDs);

                    auto currentLevelLandmarkData = _hsneAnalysisPlugin->getRoiEmbLandmarkDataDataset();
                    currentLevelLandmarkData->setData(dataLandmarks.data(), imageIDs.size(), numEnabledDimensions);
                    events().notifyDatasetDataChanged(currentLevelLandmarkData);
                }

                _updateMetaDataset = false;
            }

            });
    }

    /// Connect visual budget range actions ///
    {
        connect(&_visRangeAction, &IntegralAction::valueChanged, this, [this](const int32_t& val) {
            _visualRange = val;

            const auto currentRange = static_cast<uint32_t>(_visBudgetMaxAction.getValue() - _visBudgetMinAction.getValue());

            if (currentRange != _visualRange)
                _visBudgetMaxAction.setValue(_visBudgetMinAction.getValue() + _visualRange);

            });

        auto updateVisBudgetAction = [&]() {

            if (&_hsneAnalysisPlugin->getHsneSettingsAction() == nullptr || !_hsneAnalysisPlugin->getHsneSettingsAction().getAdvancedHsneSettingsAction().getHardCutOffAction().isChecked())
                return;

            auto numScales = _hsneAnalysisPlugin->compNumHierarchyScales();
            auto& scalesAction = _hsneAnalysisPlugin->getHsneSettingsAction().getGeneralHsneSettingsAction().getNumScalesAction();

            if (scalesAction.getValue() != numScales)
                scalesAction.setValue(numScales);
        };

        connect(&_visBudgetTargetAction, &IntegralAction::valueChanged, this, [this, updateVisBudgetAction](const int32_t& val) {
            updateVisBudgetAction();
            });
        updateVisBudgetAction();

        connect(&_visBudgetMinAction, &IntegralAction::valueChanged, this, [this](const int32_t& newMin) {
            const auto newMax = newMin + _visualRange;

            // if new max would exceed the slider boundaries, move min back
            if (newMax > _visBudgetMax)
            {
                _lockBudgetSlider = true;
                _visBudgetMinAction.setValue(_visBudgetMax - _visualRange);
            }

            // prevent endless loop
            if (_lockBudgetSlider)
            {
                _lockBudgetSlider = false;
                return;
            }

            // set new max value
            if (newMax != _visBudgetMaxAction.getValue())
            {
                _lockBudgetSlider = true;
                _visBudgetMaxAction.setValue(newMax);
            }

            });

        connect(&_visBudgetMaxAction, &IntegralAction::valueChanged, this, [this](const int32_t& newMax) {

            // if new min would exceed the slider boundaries, move max back
            if (static_cast<uint32_t>(newMax) < _visualRange)
            {
                _lockBudgetSlider = true;
                _visBudgetMaxAction.setValue(_visualRange);
            }

            const auto newMin = newMax - _visualRange;

            // prevent endless loop
            if (_lockBudgetSlider)
            {
                _lockBudgetSlider = false;
                return;
            }

            // set new min value
            if (newMin != _visBudgetMinAction.getValue())
            {
                _lockBudgetSlider = true;
                _visBudgetMinAction.setValue(newMin);
            }

            });

        connect(&_rangeHeuristicAction, &ToggleAction::toggled, this, [this](const bool& val) {
            _visBudgetMinAction.setEnabled(_rangeHeuristicAction.isChecked());
            });
    }

    /// Connect treshold for filtering landmarks ///
    {
        connect(&_landmarkFilterToggle, &ToggleAction::toggled, this, [this](const bool& val) {
            _landmarkFilterSlider.setEnabled(_landmarkFilterToggle.isChecked());
            });
    }

    /// Connect _hsneScaleUpdate ///
    {
        connect(&_hsneScaleUpdate, &HsneScaleUpdate::finished, this, [this](bool success) {

            if (success == true)
            {
                Log::info("HsneScaleWorker::finished successful");
                _updateMetaDataset = true;
                emit starttSNE();
            }
            else
                Log::warn("HsneScaleWorker::finished unsuccessful");

            });

        // Update scale level info text in UI
        connect(&_hsneScaleUpdate, &HsneScaleUpdate::scaleLevelComputed, this, &HsneScaleAction::setScale);
    }

    /// Connect _scaleUpDownActions///
    {
        // Go up and down the hierarchy for current entire view
        connect(&_scaleUpDownActions.getScaleDownAction(), &TriggerAction::triggered, this, &HsneScaleAction::refineView);
        connect(&_scaleUpDownActions.getScaleUpAction(), &TriggerAction::triggered, this, &HsneScaleAction::coarsenView);
    }

    connect(this, &HsneScaleAction::starttSNE, this, &HsneScaleAction::starttSNEAnalysis);
    connect(this, &HsneScaleAction::stoptSNE, this, &HsneScaleAction::stoptSNEAnalysis);
}

HsneScaleAction::~HsneScaleAction()
{
}

QMenu* HsneScaleAction::getContextMenu(QWidget* parent /*= nullptr*/)
{
    auto menu = new QMenu(text(), parent);

    menu->addAction(&_updateStopAction);

    return menu;
}

void HsneScaleAction::initImageSize(const QSize imgSize) {

    _inputImageSize = imgSize;
    _numImagePoints = _inputImageSize.width() * _inputImageSize.height();

    // create a vector with all global image indices
    std::vector<uint32_t> globalIDs(_numImagePoints);
    std::iota(globalIDs.begin(), globalIDs.end(), 0);

    // reshape the global indices vector into a 2d matrix
    _imageIndices = Eigen::Map<Eigen::MatrixXui>(&globalIDs[0], _inputImageSize.width(), _inputImageSize.height());

    // first region of interest is the entire image
    _roi = { { 0, 0 }, { static_cast<float>(_inputImageSize.width()), static_cast<float>(_inputImageSize.height()) } };

    // inform scale update about image size
    _hsneScaleUpdate.setImageSize(_inputImageSize);
}

void HsneScaleAction::setScale(uint32_t scale)
{
    _currentScaleLevel = scale;

    // update UI info text
    auto message = QString::number(_currentScaleLevel) + " of " + QString::number(_hsneHierarchy.getTopScale());
    _currentScaleAction.setMessage(message);   

    // Enable/Disable UI buttons for going a scale up a down
    _scaleUpDownActions.currentScaleChanged(scale);
}

void HsneScaleAction::setROI(const utils::Vector2D layerRoiBottomLeft, const utils::Vector2D layerRoiTopRight, const utils::Vector2D viewRoiXY, const utils::Vector2D viewRoiWH)
{ 
    _RoiGoodForUpdate = true;

    auto oldNumPointsInROI = utils::ROI::computeNumPixelInROI(_roi.layerBottomLeft, _roi.layerTopRight);
    auto newNumPointsInROI = utils::ROI::computeNumPixelInROI(layerRoiBottomLeft, layerRoiTopRight);

    // if full image is and has been visible, don't update
    if (oldNumPointsInROI == _numImagePoints && newNumPointsInROI == _numImagePoints)
        _RoiGoodForUpdate = false;
    // if no image is and has been visible, don't update
    if (oldNumPointsInROI == 0 && newNumPointsInROI == 0)
        _RoiGoodForUpdate = false;
    // if viewport is the same as before, don't update
    if (_roi.layerBottomLeft == layerRoiBottomLeft && _roi.layerTopRight == layerRoiTopRight)
        _RoiGoodForUpdate = false;

    _roi.layerBottomLeft = layerRoiBottomLeft;
    _roi.layerTopRight = layerRoiTopRight;
    _roi.viewRoiXY = viewRoiXY;
    _roi.viewRoiWH = viewRoiWH;

    Log::warn(fmt::format("HsneScaleAction::setROI layer {} {} {} {}", _roi.layerBottomLeft.x(), _roi.layerBottomLeft.y(), _roi.layerTopRight.x(), _roi.layerTopRight.y()));
    Log::warn(fmt::format("HsneScaleAction::setROI view {} {} {} {}", _roi.viewRoiXY.x(), _roi.viewRoiXY.y(), _roi.viewRoiWH.x(), _roi.viewRoiWH.y()));
}

void HsneScaleAction::setRefEmbExtends(utils::EmbeddingExtends extends) {
    _refEmbExtends = extends;
    auto maxString = fmt::format("x in [{:.3f}, {:.3f}], y in [{:.3f}, {:.3f}]", _refEmbExtends.x_min(), _refEmbExtends.x_max(), _refEmbExtends.y_min(), _refEmbExtends.y_max());
    Log::info("New embedding extends reference: " + maxString);
}

void HsneScaleAction::setCurrentEmbExtends(utils::EmbeddingExtends extends) {
    _currentEmbExtends = extends;

    updateEmbScaling();
}

void HsneScaleAction::updateEmbScaling()
{
    const float scaleX = (_refEmbExtends.extend_x() > 0 && _currentEmbExtends.extend_x() > 0) ? _refEmbExtends.extend_x() / _currentEmbExtends.extend_x() : 0.1;
    const float scaleY = (_refEmbExtends.extend_y() > 0 && _currentEmbExtends.extend_y() > 0) ? _refEmbExtends.extend_y() / _currentEmbExtends.extend_y() : 0.1;
    _embScaling = { scaleX * _embScalingSlider.getValue(), scaleY * _embScalingSlider.getValue() };

    auto scaleString = fmt::format("x: {} ({:.3f}), y: {} ({:.3f})", _embScaling.first, scaleX, _embScaling.second, scaleY);
    auto currentString = fmt::format("x in [{:.3f}, {:.3f}], y in [{:.3f}, {:.3f}]", _currentEmbExtends.x_min(), _currentEmbExtends.x_max(), _currentEmbExtends.y_min(), _currentEmbExtends.y_max());
    auto maxString = fmt::format("x in [{:.3f}, {:.3f}], y in [{:.3f}, {:.3f}]", _refEmbExtends.x_min(), _refEmbExtends.x_max(), _refEmbExtends.y_min(), _refEmbExtends.y_max());

    _embScaleFac.setMessage(QString::fromStdString(scaleString));
    _embCurrExt.setMessage(QString::fromStdString(currentString));
    _embMaxExt.setMessage(QString::fromStdString(maxString));

}

void HsneScaleAction::setVisualBudgetRange(const uint32_t visBudgetMin, const uint32_t visBudgetMax)
{
    if (visBudgetMax <= visBudgetMin)
        return;

    _lockBudgetSlider = true;
    _visualRange = visBudgetMax - visBudgetMin;
    _visBudgetMinAction.setValue(visBudgetMin);
    _visBudgetMaxAction.setValue(visBudgetMax);
}

void HsneScaleAction::setVisualBudgetRange(const uint32_t visBudgetMin)
{
    _lockBudgetSlider = true;
    _visBudgetMinAction.setValue(visBudgetMin);
    _visBudgetMaxAction.setValue(visBudgetMin + _visualRange);
}

utils::VisualBudgetRange HsneScaleAction::getVisualBudgetRange() const {
    return utils::VisualBudgetRange(_visBudgetMinAction.getValue(), _visBudgetMaxAction.getValue(), _visRangeAction.getValue(), _visBudgetTargetAction.getValue(), _rangeHeuristicAction.isChecked());
}

void HsneScaleAction::update()
{
    // If scale worker is still busy, don't start it again
    if (_hsneScaleUpdate.isRunning())
    {
        Log::debug("HsneScaleAction:: hsne Scale Worker is still busy");
        emit noUpdate(NoUpdate::ISRUNNING);
        return;
    }

    // see setROI() for checks: 
    // when zooming out, there should be an update but when panning with the full image in view, there should not
    // same for viewport outside image
    if (_RoiGoodForUpdate == false)
    {
        Log::debug("HsneScaleAction:: no update (e.g. same viewport, viewport change while full image visible, etc.)");
        emit noUpdate(NoUpdate::ROINOTGOODFORUPDATE);
        return;
    }

    // do not update if user chose not to
    if(_updateStopAction.isChecked())
    {
        Log::debug("HsneScaleAction:: no update (set in UI)");
        emit noUpdate(NoUpdate::SEITINUI);
        return;
    }

    Log::info("HsneScaleAction::update()");
    Log::info(fmt::format("User bottom left (width, height): {0}, {1}", _roi.layerBottomLeft.x(), _roi.layerBottomLeft.y()));
    Log::info(fmt::format("User top right   (width, height): {0}, {1}", _roi.layerTopRight.x(), _roi.layerTopRight.y()));

    // update the roi in the sequence viewer
    emit setRoiInSequenceView(_roi);

    // start the update
    computeUpdate();
}

void HsneScaleAction::computeUpdate(const utils::TraversalDirection direction /*= utils::TraversalDirection::AUTO*/)
{
    emit started();

    // If gradient descent is currently running for a previous scale update, stop it
    emit stoptSNE();

    // Deselect all items (resizing datasets with active selections might cause problems)
    _hsneAnalysisPlugin->deselectAll();

    // Get current visual budget, as defined in the UI
    const auto visualBudget = getVisualBudgetRange();

    // start worker and stop t-SNE (if is it computing in the background)
    _hsneScaleUpdate.startComputation(_embedding, _roi, _imageIndices, _idMap, _fixScaleAction.isChecked(), _tresh_influence, visualBudget, _embScaling, _currentEmbExtends,
        getLandmarkFilterNumber(), direction, _hsneAnalysisPlugin->getSelectionMapBottomToLocal(), _hsneAnalysisPlugin->getSelectionMapLocalToBottom(),
        _initEmbedding, _newTransitionMatrix);

}

void HsneScaleAction::computeTopLevelEmbedding()
{
    Log::info("HsneScaleAction::computeTopLevelEmbedding");

    // Get the top scale of the HSNE hierarchy
    const uint32_t topScaleIndex = _hsneHierarchy.getTopScale();
    const Hsne::scale_type& topScale = _hsneHierarchy.getScale(topScaleIndex);
    const uint32_t numLandmarks = topScale.size();

    // Print some debug info
    _hsneHierarchy.printScaleInfo();

    // UI set up: set max scale to go up and down manually //
    _scaleUpDownActions.setNumScales(topScaleIndex);

    // Set scale and visual range in UI based on number of landmarks in top scale: vismin = numLandmarks - (range/2)
    // DEPRECATED
    _hsneScaleUpdate.setInitalTopLevelScale(topScaleIndex);
    setScale(topScaleIndex);
    setVisualBudgetRange(numLandmarks - (static_cast<uint32_t>(getVisualBudgetRange().getRange()) / 2u));

    // set visual budget target
    _visBudgetTargetAction.setValue(10'000);

    // At first, the top level embedding contains all landmarks of the top level
    std::vector<uint32_t> localIDsOnScale(numLandmarks);
    std::iota(localIDsOnScale.begin(), localIDsOnScale.end(), 0);

    // Add ID map between local IDs and data ID
    utils::recomputeIDMap(topScale, localIDsOnScale, _idMap);

    // Add linked selection between the highest level embedding and the data (lowest/bottom level landmarks)
    _hsneHierarchy.computeSelectionMapsAtScale(topScaleIndex, localIDsOnScale, _hsneAnalysisPlugin->getSelectionMapBottomToLocal(), _hsneAnalysisPlugin->getSelectionMapLocalToBottom());

    // Get landmark data, as in InteractiveHsnePlugin::computeTSNEforLandmarks()
    std::vector<float> dataLandmarks;
    size_t numEnabledDimensions;
    {
        // Set selection linking for landmark data
        auto& mapTopLevelDataLocalToBottom = _hsneAnalysisPlugin->getSelectionMapTopLevelDataLocalToBottom();
        auto& mapTopLevelDataBottomToLocal = _hsneAnalysisPlugin->getSelectionMapTopLevelDataBottomToLocal();
        mapTopLevelDataLocalToBottom.clear();
        mapTopLevelDataLocalToBottom.resize(_idMap.size());
        mapTopLevelDataBottomToLocal.clear();
        mapTopLevelDataBottomToLocal.resize(_input->getNumPoints());

        // Get global landmark IDs
        std::vector<uint32_t> imageSelectionIDs;
        for (const auto& [dataID, embIdAndPos] : _idMap)
        {
            // add selection map entry
            mapTopLevelDataLocalToBottom[embIdAndPos.posInEmbedding].emplace_back(dataID);
            mapTopLevelDataBottomToLocal[dataID] = embIdAndPos.posInEmbedding;

            // copy data ID
            imageSelectionIDs.emplace_back(dataID);
        }
        std::sort(utils::exec_policy, imageSelectionIDs.begin(), imageSelectionIDs.end());

        // Get dimensions
        std::vector<uint32_t> enabledDimensionsIDs;
        std::tie(enabledDimensionsIDs, numEnabledDimensions) = _hsneAnalysisPlugin->enabledDimensions();

        // get data
        dataLandmarks.resize(enabledDimensionsIDs.size() * imageSelectionIDs.size());
        _input->populateDataForDimensions<std::vector<float>, std::vector<uint32_t>, std::vector<uint32_t>>(dataLandmarks, enabledDimensionsIDs, imageSelectionIDs);
        
        // set mv data set
        _topLevelLandmarkData->setData(dataLandmarks.data(), imageSelectionIDs.size(), numEnabledDimensions);
        events().notifyDatasetDataChanged(_topLevelLandmarkData);

        auto currentLevelLandmarkData = _hsneAnalysisPlugin->getRoiEmbLandmarkDataDataset();
        currentLevelLandmarkData->setData(dataLandmarks.data(), imageSelectionIDs.size(), numEnabledDimensions);
        events().notifyDatasetDataChanged(currentLevelLandmarkData);
    }

    // Initialize embedding with PCA or random
    std::vector<float> initEmbedding;
    utils::timer([&]() {
        bool pca_success = false;

        if (_hsneAnalysisPlugin->getHsneSettingsAction().getAdvancedHsneSettingsAction().getInitWithPCA().isChecked())
        {
            Log::info("HsneScaleAction::computeTopLevelEmbedding:: Compute PCA (of top level landmark data) as init embedding ");
            size_t num_comps = 2;
            auto pca_alg = convertPcaAlgorithm(_hsneAnalysisPlugin->getHsneSettingsAction().getAdvancedHsneSettingsAction().getPcaAlgorithmAction().getCurrentIndex());
            pca_success = math::pca(dataLandmarks, numEnabledDimensions, initEmbedding, num_comps, pca_alg);
            assert(initEmbedding.size() == 2ull * numLandmarks);

            if (pca_success != true)
                Log::error("HsneScaleAction::computeTopLevelEmbedding:: PCA failed. Init with random.");
        }

        if(pca_success != true)
        {
            Log::info("HsneScaleAction::computeTopLevelEmbedding:: Random init embedding... ");
            initEmbedding.resize(2ull * numLandmarks);
            auto range = utils::pyrange(static_cast<size_t>(numLandmarks));
            std::for_each(utils::exec_policy, range.begin(), range.end(), [&](const size_t i) {
                auto randomPoint = utils::randomVec(1, 1);

                initEmbedding[2u * i + 0u] = randomPoint.x;
                initEmbedding[2u * i + 1u] = randomPoint.y;
                }
            );
        }
        },
        "compute init emebdding");

    // Create Regular HSNE
    {
        // Caveat: Different selections
        // When refining a selection S0 and then selecting the refined embeddding
        // the selection S0 in the top level regular HSNE embedding will be highlighted
        // but in my top level embedding more landmarks will be highlighted
        // For now, dont try to change this. It would involve rewriting the selection mapping
        // and the image recoloring which is based on the core selection maps

        // Create scatter color data
        _regTopLevelScatterCol = mv::data().createDerivedDataset("HSNE Top Level Scatter Colors", _regHsneTopLevel, _regHsneTopLevel);
        events().notifyDatasetAdded(_regTopLevelScatterCol);
        std::vector<float> scatterColorsTopLevel(static_cast<size_t>(numLandmarks) * 3u, 0.0f);
        _regTopLevelScatterCol->setData(scatterColorsTopLevel.data(), numLandmarks, 3);
        events().notifyDatasetDataChanged(_regTopLevelScatterCol);

        RegularHsneAction* refineScaleAction = new RegularHsneAction(this, _tsneSettingsAction, _hsneHierarchy, _input, _regHsneTopLevel, _regTopLevelScatterCol, _hsneAnalysisPlugin);
        refineScaleAction->setScale(topScaleIndex);
        _regHsneTopLevel->addAction(*refineScaleAction);

        // Select the appropriate points to create a subset from
        auto selectionDataset = _input->getSelection<Points>();
        selectionDataset->indices.resize(numLandmarks);

        if (_input->isFull())
        {
            for (uint32_t i = 0; i < numLandmarks; i++)
                selectionDataset->indices[i] = topScale._landmark_to_original_data_idx[i];
        }
        else
        {
            std::vector<unsigned int> globalIndices;
            _input->getGlobalIndices(globalIndices);
            for (uint32_t i = 0; i < numLandmarks; i++)
                selectionDataset->indices[i] = globalIndices[topScale._landmark_to_original_data_idx[i]];
        }

        // Create the subset and clear the selection
        auto subset = _input->createSubsetFromSelection(QString("hsne_scale_%1").arg(topScaleIndex), nullptr, false);

        selectionDataset->indices.clear();

        _regHsneTopLevel->setSourceDataset(subset);

        // Add linked selection between the upper embedding and the bottom layer
        {
            LandmarkMap& landmarkMap = _hsneHierarchy.getInfluenceHierarchy().getMapTopDown()[topScaleIndex];

            mv::SelectionMap mapping;

            if (_input->isFull())
            {
                for (int i = 0; i < landmarkMap.size(); i++)
                {
                    int bottomLevelIdx = _hsneHierarchy.getScale(topScaleIndex)._landmark_to_original_data_idx[i];
                    mapping.getMap()[bottomLevelIdx] = landmarkMap[i];
                }
            }
            else
            {
                std::vector<unsigned int> globalIndices;
                _input->getGlobalIndices(globalIndices);
                for (int i = 0; i < landmarkMap.size(); i++)
                {
                    std::vector<unsigned int> bottomMap = landmarkMap[i];
                    for (int j = 0; j < bottomMap.size(); j++)
                    {
                        bottomMap[j] = globalIndices[bottomMap[j]];
                    }
                    int bottomLevelIdx = _hsneHierarchy.getScale(topScaleIndex)._landmark_to_original_data_idx[i];
                    mapping.getMap()[globalIndices[bottomLevelIdx]] = bottomMap;
                }
            }

            _regHsneTopLevel->addLinkedData(_input, mapping);
        }

    }

    _newTransitionMatrix = _hsneHierarchy.getTransitionMatrixAtScale(topScaleIndex);

    // just make sure to stop any ongoing t-SNE computations
    stoptSNEAnalysis();

    // Embed data with t-SNE
    Log::trace("HsneScaleAction::computeTopLevelEmbedding:: Start t-sne computation ");
    _tsneAnalysis.startComputation(_hsneAnalysisPlugin->getHsneSettingsAction().getTsneSettingsAction().getTsneParameters(), 
                                   _newTransitionMatrix, initEmbedding, numLandmarks);

    // Resize meta data sets //
    Log::debug("HsneScaleAction::computeTopLevelEmbedding:: Resize meta data sets ");

    // Mark all points in initial embedding as having been in (a non-existent) previous embedding
    std::vector<float> initialPointInitTypes(numLandmarks, utils::initTypeToFloat(utils::POINTINITTYPE::previousPos));
    _pointInitTypes->setData(initialPointInitTypes.data(), numLandmarks, 1);
    events().notifyDatasetDataChanged(_pointInitTypes);

    // All data are in ROI
    std::vector<float> initialRoiRepresentation(numLandmarks, 1.0f);
    _roiRepresentation->setData(initialRoiRepresentation.data(), numLandmarks, 1);
    events().notifyDatasetDataChanged(_roiRepresentation);

    // Get number of transition landmarks per landmarks from transition matrix
    std::vector<float> numberTransitions;
    {
        const auto& transitionMatrix = _hsneHierarchy.getTransitionMatrixAtScale(topScaleIndex);
        assert(transitionMatrix.size() == numLandmarks);
        numberTransitions.reserve(transitionMatrix.size());

        for (size_t n = 0; n < transitionMatrix.size(); ++n)
            numberTransitions.push_back(transitionMatrix[n].size());
    }

    _numberTransitions->setData(numberTransitions.data(), numberTransitions.size(), 1);
    events().notifyDatasetDataChanged(_numberTransitions);
}


void HsneScaleAction::recomputeScaleEmbedding(bool randomInitMeta)
{
    _hsneAnalysisPlugin->deselectAll();

    size_t numEmbPoints = static_cast<size_t>(_embedding->getNumPoints());

    // set init extends
    const float rad_randomMax_X = std::max(std::abs(_currentEmbExtends.x_min()), std::abs(_currentEmbExtends.x_max())) * _embScaling.first;
    const float rad_randomMax_Y = std::max(std::abs(_currentEmbExtends.y_min()), std::abs(_currentEmbExtends.y_max())) * _embScaling.second;

    Log::info(fmt::format("recomputeScaleEmbedding: Init new embedding in min/max x: {0}, y: {1} (Current extends * Scaling factor)", rad_randomMax_X, rad_randomMax_Y));

    // random init of the embedding
    auto rangeEmbPoints = utils::pyrange(numEmbPoints);
    _initEmbedding.resize(numEmbPoints * 2u);
    std::for_each(std::execution::par, rangeEmbPoints.begin(), rangeEmbPoints.end(), [&](const size_t i) {
        auto randomPoint = utils::randomVec(rad_randomMax_X, rad_randomMax_Y);

        _initEmbedding[2u * i + 0u] = randomPoint.x;
        _initEmbedding[2u * i + 1u] = randomPoint.y;
        }
    );
    _embedding->setData(_initEmbedding.data(), _embedding->getNumPoints(), 2);

    // reset all interpolated point types to random
    if (randomInitMeta)
    {
        _pointInitTypes->visitData([&](auto pointData) {
            auto range = utils::pyrange(_pointInitTypes->getNumPoints());
            std::for_each(utils::exec_policy, range.begin(), range.end(), [&](const uint i) {
                pointData[i][0] = 2.0f;
                });
            });
        events().notifyDatasetDataChanged(_pointInitTypes);
    }

    assert(_newTransitionMatrix.size() == _pointInitTypes->getNumPoints());
    assert(_newTransitionMatrix.size() == _embedding->getNumPoints());
    assert(_newTransitionMatrix.size() == _initEmbedding.size() / 2);

    // save color image as prev
    _hsneAnalysisPlugin->saveCurrentColorImageAsPrev();

    // start t-SNE **with** exaggeration (that's what the false parameter does)
    emit starttSNE(false);
}

void HsneScaleAction::publishSelectionData()
{
    Log::trace("publishSelectionData");

    // get selected data points, they are global IDs
    std::vector<uint32_t> selectionIDs = _input->getSelectionIndices();
    if (selectionIDs.size() == 0)
        return;

    std::sort(utils::exec_policy, selectionIDs.begin(), selectionIDs.end());

    Log::info(fmt::format("publishSelectionData: get {} data points", selectionIDs.size()));

    std::vector<float> data;
    std::vector<uint32_t> enabledDimensionsIDs;
    size_t numEnabledDimensions;

    // get dimensions
    std::tie(enabledDimensionsIDs, numEnabledDimensions) = _hsneAnalysisPlugin->enabledDimensions();

    // get data
    data.resize(enabledDimensionsIDs.size() * selectionIDs.size());
    _input->populateDataForDimensions<std::vector<float>, std::vector<uint32_t>, std::vector<uint32_t>>(data, enabledDimensionsIDs, selectionIDs);

    // set data in core
    auto selectionAttributeData = _hsneAnalysisPlugin->getSelectionAttributeDataDataset();
    selectionAttributeData->setData(data.data(), selectionIDs.size(), numEnabledDimensions);
    events().notifyDatasetDataChanged(selectionAttributeData);

    // Set selection linking for landmark data
    auto& mapSelectionDataLocalToBottom = _hsneAnalysisPlugin->getSelectionMapSelectionDataLocalToBottom();
    auto& mapSelectionDataBottomToLocal = _hsneAnalysisPlugin->getSelectionMapSelectionDataBottomToLocal();
    mapSelectionDataLocalToBottom.clear();
    mapSelectionDataLocalToBottom.resize(selectionIDs.size());
    mapSelectionDataBottomToLocal.clear();
    mapSelectionDataBottomToLocal.resize(_input->getNumPoints());

    // Add selection map entry
    for (uint32_t i = 0; i < selectionIDs.size(); i++)
    {
        mapSelectionDataLocalToBottom[i].emplace_back(selectionIDs[i]);
        mapSelectionDataBottomToLocal[selectionIDs[i]] = i;
    }

}


void HsneScaleAction::starttSNEAnalysis()
{
    _tsneAnalysis.stopComputation();
    
    // per default, HSNE scale embedding are computed without exaggeration here
    TsneParameters tsneParameters = _tsneSettingsAction.getTsneParameters();
    if (_noExaggerationUpdate.isChecked())
    {
        tsneParameters.setExaggerationFactor(0);
        tsneParameters.setExaggerationIter(0);
        tsneParameters.setExponentialDecayIter(0);
    }

    // Start the embedding process
    _tsneAnalysis.startComputation(tsneParameters, _newTransitionMatrix, _initEmbedding, static_cast<uint32_t>(_newTransitionMatrix.size()));
}

void HsneScaleAction::stoptSNEAnalysis()
{
    _tsneAnalysis.stopComputation();
}

void HsneScaleAction::traverseHierarchyForView(utils::TraversalDirection direction) {
    // If scale worker is still busy, don't start it again
    if (_hsneScaleUpdate.isRunning())
    {
        Log::debug("HsneScaleAction:: hsne Scale Worker is still busy");
        return;
    }

    // check if the new scale level falls within scale bounds, otherwise return
    {
        uint32_t newScaleLevel = _currentScaleLevel;
        utils::applyTraversalDirection(direction, newScaleLevel);
        
        if (newScaleLevel != std::clamp(newScaleLevel, 0u, _hsneHierarchy.getTopScale()))
        {
            Log::debug(fmt::format("HsneScaleAction::traverseHierarchyForView: new scale level {0} outside scale range [0, {1}]", newScaleLevel, _hsneHierarchy.getTopScale()));
            return;
        }
    }

    Log::info(fmt::format("HsneScaleAction::traverseHierarchyForView() go {0}", (direction == utils::TraversalDirection::UP) ? "up" : "down"));

    // start the update
    computeUpdate(direction);
}

void HsneScaleAction::refineView()
{
    Log::debug("HsneScaleAction::refineView");
    traverseHierarchyForView(utils::TraversalDirection::DOWN);
}

void HsneScaleAction::coarsenView()
{
    Log::debug("HsneScaleAction::coarsenView");
    traverseHierarchyForView(utils::TraversalDirection::UP);
}

void HsneScaleAction::compRepresents()
{
    Log::debug("HsneScaleAction::compRepresents");
    Log::warn("HsneScaleAction::compRepresents() is not yet fully implemented or tested.");
    utils::ScopedTimer compRepresentsTimer("compRepresents");

    if (_currentScaleLevel == _hsneHierarchy.getTopScale())
        return;

    // Go to uppermost scale for the current selection
    std::vector<uint32_t> selIds;
    {
        // Get selected embedding IDs
        const std::vector<uint32_t> embIds = _embedding->getSelection<Points>()->indices;

        // Get landmark ID on scale
        for (const auto& [dataID, embIdAndPosInScale] : _idMap)
        {
            for (const auto& embId : embIds)
            {
                if (embId == embIdAndPosInScale.posInEmbedding)
                    selIds.emplace_back(embIdAndPosInScale.localIdOnScale);
            }
        }

        // might as well use _hierarchy.getInfluenceHierarchy().getMapBottomUp(); ??

        // This is basically HsneHierarchy::getLocalIDsInCoarserScale but with scaleLevel starting at _currentScaleLevel and not 0

        // Get landmark ID of representative landmark on uppermost scale
        std::map<uint32_t, float> landmarkMap;
        float tresh = 0.1f;
        for (uint32_t scaleLevel = _currentScaleLevel; scaleLevel < _hsneHierarchy.getTopScale(); scaleLevel++)
        {
            _hsneHierarchy.getInfluencingLandmarksInCoarserScale(scaleLevel, selIds, landmarkMap);

            selIds.clear();
            for (const auto& [id, influence] : landmarkMap)
                if (influence > tresh)
                    selIds.emplace_back(id);
        }

        // TODO: how do I get rid of outliers?

        _firstEmbedding->getSelection<Points>()->indices.assign(selIds.cbegin(), selIds.cend());
        events().notifyDatasetDataSelectionChanged(_firstEmbedding);

    }

    // Creating a concave hull (alpha shape)
    {

    }

    // Create an Image/Pixmap from the segments
    {

    }

    // Convert QtImage to OpenCV Matrix
    {

    }

    // Compute distance transform
    {

    }

}



/// ////////////////// ///
/// GUI: Scale Up&Down ///
/// ////////////////// ///

ScaleDownUpActions::ScaleDownUpActions(QObject* parent) :
    WidgetAction(parent, "ScaleDownUpActions"),
    _scaleUpAction(this, "Up"),
    _scaleDownAction(this, "Down"),
    _numScales(0)
{
    setText("Scale");

    _scaleUpAction.setToolTip("Go a scale down");
    _scaleDownAction.setToolTip("Go a scale up");

    _scaleUpAction.setEnabled(false);
    _scaleDownAction.setEnabled(false);
}

void ScaleDownUpActions::currentScaleChanged(size_t currentScale)
{
    _scaleUpAction.setEnabled(true);
    _scaleDownAction.setEnabled(true);

    if (currentScale >= _numScales)
        _scaleUpAction.setEnabled(false);

    if (currentScale <= 0)
        _scaleDownAction.setEnabled(false);
}

ScaleDownUpActions::Widget::Widget(QWidget* parent, ScaleDownUpActions* scaleDownUpActions) :
    WidgetActionWidget(parent, scaleDownUpActions)
{
    auto layout = new QHBoxLayout();

    layout->setContentsMargins(0, 0, 0, 0);

    layout->addWidget(scaleDownUpActions->getScaleDownAction().createWidget(this));
    layout->addWidget(scaleDownUpActions->getScaleUpAction().createWidget(this));

    setLayout(layout);
}
