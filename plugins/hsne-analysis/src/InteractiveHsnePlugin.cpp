#include "InteractiveHsnePlugin.h"
#include "HsneParameters.h"
#include "HsneScaleAction.h"
#include "MeanShiftAction.h"
#include "TsneAnalysis.h"
#include "Utils.h"
#include "UtilsScale.h"
#include "ViewportSequence.h"
#include "RegularHsneAction.h"
#include "ViewportSharingActions.h"
#include "Logger.h"

#include "PointData/PointData.h"
#include "PointData/InfoAction.h"
#include "ClusterData/ClusterData.h"
#include "ImageData/Images.h"

#include <actions/PluginTriggerAction.h>

#include <QPainter>
#include <QImage>
#include <QColor>
#include <QRgb>
#include <QMap> 

#include <unordered_set>
#include <utility>
#include <algorithm>
#include <numeric>
#include <string>
#include <cmath>
#include <string>
#include <functional>
#include <limits>

Q_PLUGIN_METADATA(IID "nl.tudelft.InteractiveHsnePlugin")

using namespace mv;

/// /////////////////////// ///
/// INTERACTIVE HSNE PLUGIN ///
/// /////////////////////// ///

InteractiveHsnePlugin::InteractiveHsnePlugin(const PluginFactory* factory) :
    AnalysisPlugin(factory),
    _hierarchy(),
    _tsneROIAnalysis("ROI"),
    _tsneLandmarksAnalysis("Landmarks"),
    _hsneSettingsAction(nullptr),
    _pointInitTypes(nullptr),
    _roiRepresentation(nullptr),
    _numberTransitions(nullptr),
    _firstEmbedding(nullptr),
    _topLevelLandmarkData(nullptr),
    _roiEmbLandmarkData(nullptr),
    _selectionAttributeData(nullptr),
    _colorImgRoiHSNE(nullptr),
    _colorImgRoiHSNEprev(nullptr),
    _colorImgRoitSNE(nullptr),
    _colorImgTopLevelEmb(nullptr),
    _colorScatterTopLevelEmb(nullptr),
    _colorEmbScatBasedOnTopLevelEmb(nullptr),
    _colorImgRoiHSNEBasedOnTopLevel(nullptr),
    _colorScatterRoiHSNE(nullptr),
    _colorScatterRoitSNE(nullptr),
    _tSNEofROI(nullptr),
    _tSNEofLandmarks(nullptr),
    _initialized(false)
{
    setObjectName("InteractiveHSNE");

    // By default set to debug, setting it to "trace" will log a lot of selection info to the log file
#ifndef NDEBUG
    Log::set_level(spdlog::level::trace);
#endif // !NDEBUG

}

InteractiveHsnePlugin::~InteractiveHsnePlugin()
{
}

void InteractiveHsnePlugin::init()
{
    // Created derived dataset for embedding
    // do not use mv::data().createDerivedDataset which would connect selections
    // in the embedding with the input data set but instead
    // we manage the selection mapping here
    setOutputDataset(mv::data().createDataset<Points>("Points", "ROI embedding", getInputDataset()));

    // Get input/output datasets
    auto inputDataset = getInputDataset<Points>();
    auto outputDataset = getOutputDataset<Points>();

    events().notifyDatasetAdded(outputDataset);

    auto numPointsInput = inputDataset->getNumPoints();
    constexpr size_t numEmbeddingDimensions = 2;

    // Get image information, similar to ConvertToImagesDatasetDialog::findSourceImagesDataset in ImageViewerPlugin
    // Assume that the first child of the input dataset that is an image is the correct image layout
    {
        mv::Dataset<Images> inputdataImage = nullptr;
        for (auto childHierarchyItem : inputDataset->getDataHierarchyItem().getChildren()) {
            // Get image dimensions in case of an images dataset
            if (childHierarchyItem->getDataType() == ImageType) {
                inputdataImage = childHierarchyItem->getDataset();
                break;
            }
        }

        if (!inputdataImage.isValid())
        {
            Log::error("InteractiveHsnePlugin::init: Error. Data set is not (connected to) an image");
            return;
        }

        _inputImageSize = inputdataImage->getImageSize();
        _inputImageLoadPath = QFileInfo(inputdataImage->getImageFilePaths().first()).dir().absolutePath().toStdString();

        if (static_cast<uint64_t>(_inputImageSize.width()) * _inputImageSize.height() >= std::numeric_limits<uint32_t>::max())
            Log::error("InteractiveHsnePlugin::init: Error. Image is too large to be indexed.");
        if (static_cast<uint64_t>(numPointsInput) * inputDataset->getNumDimensions() >= std::numeric_limits<uint32_t>::max())
            Log::error("InteractiveHsnePlugin::init: Error. Data is too large to be indexed");
    }

    // helper function to set up meta data set
    auto setupMetaDataset = [this, numPointsInput](Dataset<Points>& dataset, std::vector<float>& initData, std::string identifier, uint32_t dims, Dataset<Points>& sourceDataset) {
        dataset = mv::data().createDerivedDataset<Points>(QString::fromStdString(identifier), sourceDataset);
        events().notifyDatasetAdded(dataset);
        dataset->setData(initData.data(), numPointsInput, dims);    // change this after the hierarchy is initialized, specifically data size
        events().notifyDatasetDataChanged(dataset);

    };

    // helper function to set up meta image data set
    // Derive a dataset from the input which will mirror the embedding
    // and is used for image recoloring based on the embedding
    // Update the dataset when the gradient descent is finished
    auto setupColorMappingDataset = [this, numPointsInput](Dataset<Points>& dataset, std::string identifier, Dataset<Points>& UI_parent) {
        dataset = mv::data().createDataset<Points>("Points", QString::fromStdString("Recolored Img " + identifier), UI_parent);
        events().notifyDatasetAdded(dataset);

        constexpr uint32_t numColorChannels = 3u;

        std::vector<float> initialColorMappingData(static_cast<size_t>(numPointsInput) * numColorChannels, 0);
        dataset->setData(initialColorMappingData.data(), numPointsInput, numColorChannels);
        events().notifyDatasetDataChanged(dataset);

        // Derive image data from mirrored embedding
        auto colorMappingImage = mv::data().createDataset<Images>("Images", QString::fromStdString(identifier + "Image"), dataset);

        colorMappingImage->setType(ImageData::Type::Stack);
        colorMappingImage->setNumberOfImages(numColorChannels);
        colorMappingImage->setImageSize(_inputImageSize);
        colorMappingImage->setNumberOfComponentsPerPixel(1);

        events().notifyDatasetAdded(colorMappingImage);

    };

    // Set the output dataset (embedding) size
    {
        std::vector<float> initialData(static_cast<size_t>(numPointsInput) * numEmbeddingDimensions, 0.0f);
        outputDataset->setData(initialData.data(), numPointsInput, numEmbeddingDimensions);
        events().notifyDatasetDataChanged(outputDataset);

        // Save the first top scale embedding permanently
        _firstEmbedding = mv::data().createDataset<Points>("Points", "First Top Level Embedding", inputDataset);
        _firstEmbedding->setProperty("Init", false);
        events().notifyDatasetAdded(_firstEmbedding);
        _firstEmbedding->setData(initialData.data(), numPointsInput, numEmbeddingDimensions);
        events().notifyDatasetDataChanged(_firstEmbedding);

        // Top level HSNE embedding
        //_regHsneTopLevel = mv::data().createDataset<Points>("Points", "HSNE Top Level", inputDataset);
        _regHsneTopLevel = mv::data().createDerivedDataset("HSNE Top Level", inputDataset, inputDataset);
        events().notifyDatasetAdded(_regHsneTopLevel);
        _regHsneTopLevel->setData(initialData.data(), numPointsInput, numEmbeddingDimensions);
        events().notifyDatasetDataChanged(_regHsneTopLevel);

    }
    
    // Set initial top level landmark data set
    {
        // init attribute data
        std::vector<float> initialData(static_cast<size_t>(1) * inputDataset->getNumDimensions(), 0.0f);

        // set init top level landmark attribute data
        _topLevelLandmarkData = mv::data().createDataset<Points>("Points", "Top Level Landmark Data", _firstEmbedding);
        events().notifyDatasetAdded(_topLevelLandmarkData);
        _topLevelLandmarkData->setData(initialData.data(), 1, numEmbeddingDimensions);
        events().notifyDatasetDataChanged(_topLevelLandmarkData);

        // set current level data 
        _roiEmbLandmarkData = mv::data().createDataset<Points>("Points", "Current ROI Emb Landmark Data", outputDataset);
        events().notifyDatasetAdded(_roiEmbLandmarkData);
        _roiEmbLandmarkData->setData(initialData.data(), 1, numEmbeddingDimensions);
        events().notifyDatasetDataChanged(_roiEmbLandmarkData);

        _selectionAttributeData = mv::data().createDataset<Points>("Points", "Selection Data", inputDataset);
        events().notifyDatasetAdded(_selectionAttributeData);
        _selectionAttributeData->setData(initialData.data(), 1, numEmbeddingDimensions);
        events().notifyDatasetDataChanged(_selectionAttributeData);       
    }

    // t-SNE data sets
    {
        // Derive a dataset from input which will be a t-SNE embedding of the ROI
        _tSNEofROI = mv::data().createDataset<Points>("Points", "t-SNE ROI", inputDataset);
        _tSNEofROI->setProperty("Init", false);
        events().notifyDatasetAdded(_tSNEofROI);

        std::vector<float> tSNEofROIData(numPointsInput * numEmbeddingDimensions, 0.0f);
        _tSNEofROI->setData(tSNEofROIData.data(), numPointsInput, numEmbeddingDimensions);
        events().notifyDatasetDataChanged(_tSNEofROI);

        // recoloring for scatter plot
        std::vector<float> scatterColorstSNE;
        scatterColorstSNE.resize(static_cast<size_t>(numPointsInput) * 3u, 0.0f);
        
        _imgColorstSNE.resize(static_cast<size_t>(numPointsInput) * 3u, 0.0f);

        _colorScatterRoitSNE = mv::data().createDerivedDataset<Points>("Scatter colors (t-SNE)", _tSNEofROI);
        events().notifyDatasetAdded(_colorScatterRoitSNE);
        _colorScatterRoitSNE->setData(scatterColorstSNE.data(), numPointsInput, 3);    // change this after the hierarchy is initialized, specifically data size
        events().notifyDatasetDataChanged(_colorScatterRoitSNE);

        // Derive a dataset from input which will be a t-SNE embedding of the landmarks
        // the _tSNEofLandmarks data set should of course be smaller than numPointsInput but at this
        // point we do not know the number of landmarks yet
        _tSNEofLandmarks = mv::data().createDataset<Points>("Points", "t-SNE ROI (Landmarks)", inputDataset);
        _tSNEofLandmarks->setProperty("Init", false);
        events().notifyDatasetAdded(_tSNEofLandmarks);

        std::vector<float> tSNEofLandmarks;
        tSNEofLandmarks.resize(numPointsInput * numEmbeddingDimensions);
        _tSNEofLandmarks->setData(tSNEofLandmarks.data(), numPointsInput, numEmbeddingDimensions);
        events().notifyDatasetDataChanged(_tSNEofLandmarks);
    }

    // Meta data sets
    {
        _imgColorsRoiHSNE.resize(static_cast<size_t>(numPointsInput) * 3u, 0.0f);
        _imgColorsTopLevelEmb.resize(static_cast<size_t>(numPointsInput) * 3u, 0.0f);

        // initial uniform data
        std::vector<float> initialPointInitTypes(numPointsInput, utils::initTypeToFloat(utils::POINTINITTYPE::previousPos));
        std::vector<float> initialRoiRepresentation(numPointsInput, 1.0f);

        // Derive a dataset from the output which will be used to recolor each point according to 
        // it's initialization role (previous position, interpolated or random)
        setupMetaDataset(_pointInitTypes, initialPointInitTypes, "Point Init Types", 1, outputDataset);

        // Derive a dataset from the output which will be used to resize each point according to 
        // it's roi representation: 1 -> a landmark represents only pixel inside the roi, 0 -> only outside roi
        setupMetaDataset(_roiRepresentation, initialRoiRepresentation, "ROI Representation", 1, outputDataset);

        // Derive a dataset from the output which will inform about the number of transition values
        // per landmark in the current embedding on the current scale
        setupMetaDataset(_numberTransitions, initialRoiRepresentation, "Number Transitions", 1, outputDataset);

        // Data set for coloring the scatterplot, here is holds rgb colors [0, 255] as sampled from a colormap
        setupMetaDataset(_colorScatterRoiHSNE, _imgColorsRoiHSNE, "Scatter colors", 3, outputDataset);

        // Data set for coloring the top level embedding scatterplot, here is holds rgb colors [0, 255] as sampled from a colormap
        setupMetaDataset(_colorScatterTopLevelEmb, _imgColorsTopLevelEmb, "Top Level Emb scatter colors", 3, _firstEmbedding);

        // Data set for coloring the top level embedding scatterplot, here is holds rgb colors [0, 255] as sampled from a colormap
        setupMetaDataset(_colorEmbScatBasedOnTopLevelEmb, _imgColorsTopLevelEmb, "Emb coloring based on top level", 3, outputDataset);

        // Add cluster data set of top level embedding
        _topLevelEmbClusters = mv::data().createDataset<Clusters>("Cluster", "Top level Emb clusters", _firstEmbedding);
        events().notifyDatasetAdded(_topLevelEmbClusters);

        // set up recolored image data sets
        setupColorMappingDataset(_colorImgRoiHSNE, "Hsne ROI", outputDataset);
        setupColorMappingDataset(_colorImgRoiHSNEprev, "Hsne ROI (previous)", outputDataset);
        setupColorMappingDataset(_colorImgRoiHSNEBasedOnTopLevel, "Hsne ROI (based on Top Level Emb)", outputDataset);
        setupColorMappingDataset(_colorImgRoitSNE, "t-SNE ROI", _tSNEofROI);
        setupColorMappingDataset(_colorImgTopLevelEmb, "Top Level Emb", _firstEmbedding);
    }

    // set up selection locks
    for (auto& dataset : { inputDataset, outputDataset, _firstEmbedding, _roiEmbLandmarkData, _selectionAttributeData, _regHsneTopLevel,
        _topLevelLandmarkData, _tSNEofROI, _tSNEofLandmarks, _colorImgRoiHSNE, _colorImgRoiHSNEprev, _colorImgRoitSNE, _colorImgTopLevelEmb })
        _selectionLocks[dataset->getId().toStdString()] = utils::CyclicLock(2);
    
    // Create new HSNE settings actions
    _hsneSettingsAction = std::make_shared<HsneSettingsAction>(this);

    // Some access helper
    auto& hsneScaleAction = _hsneSettingsAction->getInteractiveScaleAction();
    auto& viewportAction = _hsneSettingsAction->getViewportSequenceAction();
    //auto& imageViewportSharingAction = _hsneSettingsAction->getHsneImageViewportSharingAction(); // REMOVE
    auto& generalTsneSettings = _hsneSettingsAction->getTsneSettingsAction().getGeneralTsneSettingsAction();

    // Set the image size in the interactive scale action and use entire image as first roi
    hsneScaleAction.initImageSize(_inputImageSize);

    // No need to init this any more - done when connecting to image viewer
    //_layerRoiBottomLeft = { 0, 0 };
    //_layerRoiTopRight = { static_cast<float>(_inputImageSize.width()), static_cast<float>(_inputImageSize.height()) };
    //viewportAction.appendROI({ _layerRoiBottomLeft, _layerRoiTopRight });

    // Set the default number of hierarchy scales based on number of points
    const uint32_t numHierarchyScales = compNumHierarchyScales();
    _hsneSettingsAction->getGeneralHsneSettingsAction().getNumScalesAction().setValue(numHierarchyScales);

    // Set dimension selection data
    _hsneSettingsAction->getDimensionSelectionAction().getPickerAction().setPointsDataset(inputDataset);

    // Set embedding data set names in t-SNE settings UI
    {
        QMap<QString, QString> embDatasets; 

        for (const auto& data : { outputDataset, _tSNEofROI,  _tSNEofLandmarks } )
            embDatasets[data->getGuiName()] = data->getId();

        generalTsneSettings.setEmbDatasets(embDatasets);
    }

    // Add actions to UI
    outputDataset->addAction(_hsneSettingsAction->getGeneralHsneSettingsAction());
    outputDataset->addAction(_hsneSettingsAction->getAdvancedHsneSettingsAction());
    outputDataset->addAction(_hsneSettingsAction->getInteractiveScaleAction());
    outputDataset->addAction(_hsneSettingsAction->getTsneSettingsAction().getGeneralTsneSettingsAction());  // t-SNE settings
    outputDataset->addAction(_hsneSettingsAction->getViewportSequenceAction());
    outputDataset->addAction(_hsneSettingsAction->getDimensionSelectionAction());

    _hsneSettingsAction->getMeanShiftActionAction().expand();
    _firstEmbedding->addAction(_hsneSettingsAction->getMeanShiftActionAction());

    // Focus on output dataset in UI (which also shows the plugin UI)
    inputDataset->getDataHierarchyItem().setExpanded(true);
    outputDataset->getDataHierarchyItem().select();
    outputDataset->getDataHierarchyItem().setExpanded(true);

    // Do not show data info by default to give more space to other settings
    outputDataset->_infoAction->collapse();

    // Helper: Set some UI info during t-SNE computation and get embedding 
    auto connectMetaTsne = [this](TsneAnalysis& analysis, Dataset<Points>& dataset) -> void
    {
        connect(&analysis, &TsneAnalysis::finished, this, [this, &dataset]() {
            _hsneSettingsAction->getTsneSettingsAction().setReadOnly(false);
            });

        connect(&analysis, &TsneAnalysis::embeddingUpdate, this, [this, &dataset](const std::vector<float>& emb, const uint32_t& numPoints, const uint32_t& numDimensions) {

            dataset->setData(emb, numDimensions);

            events().notifyDatasetDataChanged(dataset);
            });

    };

    /// Connect _tsneROIAnalysis ///
    {
        // Update the color map data set when ROI t-SNE is finished
        connect(&_tsneROIAnalysis, &TsneAnalysis::finished, this, &InteractiveHsnePlugin::setColorMapDataRoitSNE);

        // Update the color map every 100 iterations
        connect(&_tsneROIAnalysis, &TsneAnalysis::embeddingUpdate, this, [this](const std::vector<float>& emb, const uint32_t& numPoints, const uint32_t& numDimensions){
            if (_tsneROIAnalysis.getNumIterations() % 100 == 0)
                setColorMapDataRoitSNE();
            });

        connectMetaTsne(_tsneROIAnalysis, _tSNEofROI);
    }

    /// Connect _tsneLandmarksAnalysis ///
    {
        connectMetaTsne(_tsneLandmarksAnalysis, _tSNEofLandmarks);
    }

    /// Connect _hsneSettingsAction ///
    {
        connect(&_hsneSettingsAction->getGeneralHsneSettingsAction().getInitAction(), &TriggerAction::triggered, this, [this, &hsneScaleAction](bool toggled) {
            _hsneSettingsAction->setReadOnly(true);

            std::vector<bool> enabledDimensions = _hsneSettingsAction->getDimensionSelectionAction().getPickerAction().getEnabledDimensions();

            // Initialize the HSNE algorithm with the given parameters
            _hierarchy.initialize(_core, *getInputDataset<Points>(), enabledDimensions, _hsneSettingsAction->getHsneParameters(), _inputImageLoadPath);

            // Compute top-level embedding
            hsneScaleAction.computeTopLevelEmbedding();
            _initialized = true;

            });

        connect(&_hsneSettingsAction->getTsneSettingsAction().getComputationAction().getContinueComputationAction(), &TriggerAction::triggered, this, [this]() {
            _hsneSettingsAction->getTsneSettingsAction().setReadOnly(true);

            continueComputation();
            });

        connect(&_hsneSettingsAction->getTsneSettingsAction().getComputationAction().getStopComputationAction(), &TriggerAction::triggered, this, [this]() {
            stopComputation();
            });

        connect(&_hsneSettingsAction->getGeneralHsneSettingsAction().getTSNERoiAction(), &TriggerAction::triggered, this, &InteractiveHsnePlugin::computeTSNEforROI);
        connect(&_hsneSettingsAction->getGeneralHsneSettingsAction().getTSNELandmarkAction(), &TriggerAction::triggered, this, &InteractiveHsnePlugin::computeTSNEforLandmarks);
    
        connect(&_hsneSettingsAction->getInteractiveScaleAction().getColorMapRoiEmbAction(), &ColorMapAction::imageChanged, this, [this](const QImage& image) {
            setColorMapDataRoiHSNE(); 
            setColorMapDataRoitSNE();
            });

    }

    // add ROI to sequence view when update is performed
    connect(&hsneScaleAction, &HsneScaleAction::setRoiInSequenceView, &viewportAction, &ViewportSequence::appendROI);

    // connect mean-shift analysis of top level embedding
    connect(&_hsneSettingsAction->getMeanShiftActionAction().getUseClusterColorsAction(), &ToggleAction::toggled, this, &InteractiveHsnePlugin::setColorMapDataTopLevelEmb);
    connect(&_hsneSettingsAction->getMeanShiftActionAction(), &MeanShiftAction::newClusterColors, this, &InteractiveHsnePlugin::setColorMapDataTopLevelEmb);

    // connect top level recoloring
    connect(&hsneScaleAction.getColorMapFirstEmbAction(), &ColorMapAction::imageChanged, this, [this](const QImage& image) {setColorMapDataTopLevelEmb(); });

    // make sure that the viewport updates correctly after setting no update in UI, moving backwards, setting do update in UI again and then navigating in the image 
    connect(&hsneScaleAction, &HsneScaleAction::noUpdate, this, [&viewportAction](const NoUpdate& reason) {
        if (reason == NoUpdate::SEITINUI || reason == NoUpdate::ROINOTGOODFORUPDATE)
            viewportAction.setLockedAddRoi(false);
        });

    // Connect viewport update signal from image viewer (if connected)
    connect(&viewportAction.getViewportSharingActions(), &ViewportSharingActions::viewportChanged, this, &InteractiveHsnePlugin::updateImageViewport);

    // update embedding when user changed viewport from sequence list
    connect(&viewportAction, &ViewportSequence::updatedROIInSequenceView, this, [this](const utils::ROI& roi) {
        const auto layerRoiBottomLeft = QVector3D(roi.layerBottomLeft.x(), roi.layerBottomLeft.y(), 0.f);
        const auto layerRoiTopRight = QVector3D(roi.layerTopRight.x(), roi.layerTopRight.y(), 0.f);

        const auto viewRoiXY = QVector3D(roi.viewRoiXY.x(), roi.viewRoiXY.y(), 0.f);
        const auto viewRoiWH = QVector3D(roi.viewRoiWH.x(), roi.viewRoiWH.y(), 0.f);

        updateImageViewport(layerRoiBottomLeft, layerRoiTopRight, viewRoiXY, viewRoiWH);
        });

    // Connect/Handle data events
    {
        // Connect selection mappings
        //      The selection mapping is handled here, outside the core, since the core does not (afaik) support the kind of
        //      one-to-multiple and multiple-to-one mapping between, datasets of different sizes
        connect(&_input[0], &Dataset<DatasetImpl>::dataSelectionChanged, this, &InteractiveHsnePlugin::onSelectionInImage);
        connect(&_output[0], &Dataset<DatasetImpl>::dataSelectionChanged, this, &InteractiveHsnePlugin::onSelectionInEmbedding);
        connect(&_tSNEofROI, &Dataset<Points>::dataSelectionChanged, this, &InteractiveHsnePlugin::onSelectionInROItSNE);
        connect(&_tSNEofLandmarks, &Dataset<Points>::dataSelectionChanged, this, &InteractiveHsnePlugin::onSelectionInLandmarktSNE);
        connect(&_colorImgRoiHSNE, &Dataset<Points>::dataSelectionChanged, this, [this]() {onSelectionInColorMappingHsneRoi(_colorImgRoiHSNE); });
        connect(&_colorImgRoiHSNEprev, &Dataset<Points>::dataSelectionChanged, this, [this]() {onSelectionInColorMappingHsneRoi(_colorImgRoiHSNEprev); });
        connect(&_colorImgRoitSNE, &Dataset<Points>::dataSelectionChanged, this, &InteractiveHsnePlugin::onSelectionInColorMappingtSNERoi);
        connect(&_firstEmbedding, &Dataset<Points>::dataSelectionChanged, this, &InteractiveHsnePlugin::onSelectionFirstEmbedding);
        connect(&_topLevelLandmarkData, &Dataset<Points>::dataSelectionChanged, this, &InteractiveHsnePlugin::onSelectionFirstEmbeddingData);
        connect(&_roiEmbLandmarkData, &Dataset<Points>::dataSelectionChanged, this, &InteractiveHsnePlugin::onSelectionCurrentLevelLandmarkData);
        connect(&_selectionAttributeData, &Dataset<Points>::dataSelectionChanged, this, &InteractiveHsnePlugin::onSelectionSelectionLandmarkData);
        connect(&_colorImgTopLevelEmb, &Dataset<Points>::dataSelectionChanged, this, &InteractiveHsnePlugin::onSelectionTopLevelImage);
        //connect(&_regHsneTopLevel, &Dataset<Points>::dataSelectionChanged, this, &InteractiveHsnePlugin::onSelectionRegHsneTopLevelEmbedding);        

        // Update dimension selection with new data
        connect(&_input[0], &Dataset<DatasetImpl>::dataChanged, this, [this, inputDataset]() {
            _hsneSettingsAction->getDimensionSelectionAction().getPickerAction().setPointsDataset(inputDataset);

            Log::warn("Dataset::dataChanged: changing number of data points will probably result in faulty results or errors");
            });

    }

}

void InteractiveHsnePlugin::updateImageViewport(const QVector3D layerRoiBottomLeft, const QVector3D layerRoiTopRight, const QVector3D viewRoiXY, const QVector3D viewRoiWH)
{
    if (!_initialized)
        return;

    // clamp to image height and width
    auto clampVec = [this](const QVector3D& roi) -> utils::Vector2D {
        return utils::Vector2D(std::clamp(static_cast<int>(std::round(roi.x())), 0, _inputImageSize.width()),
            std::clamp(static_cast<int>(std::round(roi.y())), 0, _inputImageSize.height()));
    };

    _layerRoiBottomLeft = clampVec(layerRoiBottomLeft);
    _layerRoiTopRight = clampVec(layerRoiTopRight);

    _viewRoiXY = { viewRoiXY.x(), viewRoiXY.y() };
    _viewRoiWH = { viewRoiWH.x(), viewRoiWH.y() };

    // set roi extends
    _hsneSettingsAction->getInteractiveScaleAction().setROI(_layerRoiBottomLeft, _layerRoiTopRight, _viewRoiXY, _viewRoiWH);

    // start an update embedding in a different thread
    _hsneSettingsAction->getInteractiveScaleAction().update();
}

void InteractiveHsnePlugin::selectionMapping(const mv::Dataset<Points> selectionInputData, const LandmarkMap& selectionMap, mv::Dataset<Points> selectionOutputData, utils::CyclicLock& lock) {
    lock++;

    Log::trace(fmt::format("selectionMapping from {0} to {1}", selectionInputData->getGuiName().toStdString(), selectionOutputData->getGuiName().toStdString()));
    Log::trace(fmt::format("selectionMapping lock {0}", (lock.isLocked() ? "locked" : "unlocked")));

    // to prevent infinite selection loops the locks cycles through locked and unlocked stages
    if (lock.isLocked())
        return;

    // if there is nothing to be mapped, don't do anything
    if (selectionMap.size() == 0)
        return;
    
    // "Selection map is supposed to be of the same size as the selection input data
    assert(selectionMap.size() == selectionInputData->getNumPoints());

    const mv::Dataset<Points>& selectionInput = selectionInputData->getSelection<Points>();
    auto& selectionOutputIndx = selectionOutputData->getSelection<Points>()->indices;

    // to ensure only unique elements, sort and call std::unique 
    std::vector<uint32_t> selectionIndices;

    for (const auto selectionIndex : selectionInput->indices)
    {
        if (selectionMap[selectionIndex].empty())
            continue;
            
        selectionIndices.insert(selectionIndices.end(), selectionMap[selectionIndex].begin(), selectionMap[selectionIndex].end());
    }

    std::sort(utils::exec_policy, selectionIndices.begin(), selectionIndices.end());
    auto last = std::unique(utils::exec_policy, selectionIndices.begin(), selectionIndices.end());
    selectionIndices.erase(last, selectionIndices.end());

    Log::trace("Publish selection");
    selectionOutputIndx = std::move(selectionIndices);

    events().notifyDatasetDataSelectionChanged(selectionOutputData);
}

void InteractiveHsnePlugin::selectionMapping(const mv::Dataset<Points> selectionInputData, const LandmarkMapSingle& selectionMap, mv::Dataset<Points> selectionOutputData, utils::CyclicLock& lock) {
    lock++;

    Log::trace(fmt::format("selectionMapping from {0} to {1}", selectionInputData->getGuiName().toStdString(), selectionOutputData->getGuiName().toStdString()));
    Log::trace(fmt::format("selectionMapping lock {0}", (lock.isLocked() ? "locked" : "unlocked")));

    // to prevent infinite selection loops the locks cycles through locked and unlocked stages
    if (lock.isLocked())
        return;

    // if there is nothing to be mapped, don't do anything
    if (selectionMap.size() == 0)
        return;
    
    // "Selection map is supposed to be of the same size as the selection input data
    assert(selectionMap.size() == selectionInputData->getNumPoints());

    const mv::Dataset<Points>& selectionInput = selectionInputData->getSelection<Points>();
    auto& selectionOutputIndx = selectionOutputData->getSelection<Points>()-> indices;

    // to ensure only unique elements, sort and call std::unique 
    std::vector<uint32_t> selectionIndices;

    // For all selected indices in the embedding, look up to which bottom level IDs they correspond
    for (const auto selectionIndex : selectionInput->indices)
    {
        if (selectionMap[selectionIndex] == std::numeric_limits<uint32_t>::max())
            continue;

        selectionIndices.insert(selectionIndices.end(), selectionMap[selectionIndex]);
    }

    std::sort(utils::exec_policy, selectionIndices.begin(), selectionIndices.end());
    auto last = std::unique(utils::exec_policy, selectionIndices.begin(), selectionIndices.end());
    selectionIndices.erase(last, selectionIndices.end());

    Log::trace("Publish selection");
    selectionOutputIndx = std::move(selectionIndices);
    events().notifyDatasetDataSelectionChanged(selectionOutputData);
}

void InteractiveHsnePlugin::onSelectionInEmbedding() {
    Log::trace("onSelectionInEmbedding");
    // Selection in embedding maps to selection in image using _mappingLocalToBottom
    selectionMapping(getOutputDataset<Points>(), _mappingLocalToBottom, getInputDataset<Points>(), _selectionLocks[_input[0]->getId().toStdString()]);
}

void InteractiveHsnePlugin::onSelectionFirstEmbedding() {
    Log::trace("onSelectionFirstEmbedding");
    // Selection in first top scale embedding maps to selection in image using _topLevelEmbMapLocalToBottom
    selectionMapping(_firstEmbedding, _topLevelEmbMapLocalToBottom, getInputDataset<Points>(), _selectionLocks[_firstEmbedding->getId().toStdString()]);
    
    // Selection in first top scale embedding maps to selection in top level image recolor
//    selectionMapping(_firstEmbedding, _topLevelEmbMapLocalToBottom, _colorImgTopLevelEmb, _selectionLocks[_firstEmbedding->getId().toStdString()]);
    selectionMapping(_firstEmbedding, _topLevelEmbMapLocalToBottom, _colorImgTopLevelEmb, _selectionLocks[_colorImgTopLevelEmb->getId().toStdString()]);

}

void InteractiveHsnePlugin::onSelectionFirstEmbeddingData() {
    Log::trace("onSelectionFirstEmbeddingData");
    // Selection in first top scale landmark data maps to selection in image using _topLevelDataMapLocalToBottom
    selectionMapping(_topLevelLandmarkData, _topLevelDataMapLocalToBottom, getInputDataset<Points>(), _selectionLocks[_topLevelLandmarkData->getId().toStdString()]);
}

void InteractiveHsnePlugin::onSelectionCurrentLevelLandmarkData() {
    Log::trace("onSelectionCurrentLevelLandmarkData");
    // Selection in first top scale landmark data maps to selection in image using _currentLevelDataMapLocalToBottom
    selectionMapping(_roiEmbLandmarkData, _currentLevelDataMapLocalToBottom, getInputDataset<Points>(), _selectionLocks[_roiEmbLandmarkData->getId().toStdString()]);
}

void InteractiveHsnePlugin::onSelectionSelectionLandmarkData() {
    Log::trace("onSelectionSelectionLandmarkData");
    // Selection in first top scale landmark data maps to selection in image using _currentLevelDataMapLocalToBottom
    selectionMapping(_selectionAttributeData, _selectionAttributeDataMapLocalToBottom, getInputDataset<Points>(), _selectionLocks[_selectionAttributeData->getId().toStdString()]);
}

void InteractiveHsnePlugin::onSelectionTopLevelImage() {
    Log::trace("onSelectionTopLevelImage");
    // Selection in first top scale landmark data maps to selection in the recolored image based on the top level embedding back to the top level embedding
    selectionMapping(_colorImgTopLevelEmb, _topLevelEmbMapBottomToLocal, _firstEmbedding, _selectionLocks[_colorImgTopLevelEmb->getId().toStdString()]);
}

void InteractiveHsnePlugin::onSelectionRegHsneTopLevelEmbedding(){
    Log::trace("onSelectionRegHsneTopLevelEmbedding");

    auto& lock = _selectionLocks[_regHsneTopLevel->getId().toStdString()];
    lock++;
    if (lock.isLocked())
        return;

    // copy the selection sicne the embeddings are identical 
    _firstEmbedding->getSelection<Points>()->indices = _regHsneTopLevel->getSelection<Points>()->indices;
    events().notifyDatasetDataSelectionChanged(_firstEmbedding);

}

void InteractiveHsnePlugin::onSelectionInImage() {
    Log::trace("onSelectionInImage");
    auto inputData = getInputDataset<Points>();

    // Selection in image maps to selection in hsne embedding 
    selectionMapping(inputData, _mappingBottomToLocal, getOutputDataset<Points>(), _selectionLocks[_input[0]->getId().toStdString()]);

    // Selection in first top scale embedding
    selectionMapping(inputData, _topLevelEmbMapBottomToLocal, _firstEmbedding, _selectionLocks[_firstEmbedding->getId().toStdString()]);

    // Selection in first top scale regular HSNE embedding, reuse the selection from first top scale embedding
    //{
    //    auto& lock = _selectionLocks[_firstEmbedding->getId().toStdString()];
    //    lock++;
    //    if (!lock.isLocked())
    //    {
    //        _regHsneTopLevel->getSelection<Points>()->indices = _firstEmbedding->getSelection<Points>()->indices;
    //        events().notifyDatasetDataSelectionChanged(_regHsneTopLevel);
    //    }
    //}

    // Selection in first top scale landmark data
    selectionMapping(inputData, _topLevelDataMapBottomToLocal, _topLevelLandmarkData, _selectionLocks[_topLevelLandmarkData->getId().toStdString()]);

    // Selection in current scale landmark data
    selectionMapping(inputData, _currentLevelDataMapBottomToLocal, _roiEmbLandmarkData, _selectionLocks[_roiEmbLandmarkData->getId().toStdString()]);

    // Selection in selection attribute data
    selectionMapping(inputData, _selectionAttributeDataMapBottomToLocal, _selectionAttributeData, _selectionLocks[_selectionAttributeData->getId().toStdString()]);

    // Selection in image maps to selection in ROI tSNE 
    if (_tSNEofROI->getProperty("Init").toBool())
        selectionMapping(inputData, _mappingImageToROItSNE, _tSNEofROI, _selectionLocks[_tSNEofROI->getId().toStdString()]);

    // Selection in image maps to selection in landmark tSNE 
    if (_tSNEofLandmarks->getProperty("Init").toBool())
        selectionMapping(inputData, _mappingImageToLandmarktSNE, _tSNEofLandmarks, _selectionLocks[_tSNEofLandmarks->getId().toStdString()]);

    // Selection in input maps to selection in color images
    for (auto& dataset : { _colorImgRoiHSNE , _colorImgRoiHSNEprev, _colorImgRoitSNE })
    {
        _selectionLocks[dataset->getId().toStdString()].lock();
        const auto& sel = inputData->getSelection<Points>()->indices;
        dataset->getSelection<Points>()->indices.assign(sel.cbegin(), sel.cend());
        events().notifyDatasetDataSelectionChanged(dataset);
    }

}

void InteractiveHsnePlugin::onSelectionInROItSNE() {
    if (!_tSNEofROI->getProperty("Init").toBool())
        return;
    
    Log::trace("onSelectionInROItSNE");

    // Selection in ROI t-SNE maps to image data
    selectionMapping(_tSNEofROI, _mappingROItSNEtoImage, getInputDataset<Points>(), _selectionLocks[_tSNEofROI->getId().toStdString()]);
}

void InteractiveHsnePlugin::onSelectionInLandmarktSNE() {
    if (!_tSNEofLandmarks->getProperty("Init").toBool())
        return;

    Log::trace("onSelectionInLandmarktSNE");

    // Selection in landmark t-SNE maps to image data
    selectionMapping(_tSNEofLandmarks, _mappingLandmarktSNEtoImage, getInputDataset<Points>(), _selectionLocks[_tSNEofLandmarks->getId().toStdString()]);
}

void InteractiveHsnePlugin::onSelectionInColorMappingHsneRoi(Dataset<Points> colorImgRoiHSNE) {
    auto& lock = _selectionLocks[colorImgRoiHSNE->getId().toStdString()];
    lock++;
    if (lock.isLocked())
        return;

    Log::trace("onSelectionInColorMappingHsneRoi: " + colorImgRoiHSNE->getGuiName().toStdString());

    // Selection in color map (overlayed on image) maps to image data
    // no call to selectionMapping() since we can just copy the same selection IDs
    const auto& sel = colorImgRoiHSNE->getSelection<Points>()->indices;
    auto inputData = getInputDataset<Points>();
    inputData->getSelection<Points>()->indices.assign(sel.cbegin(), sel.cend());
    events().notifyDatasetDataSelectionChanged(inputData);
}

void InteractiveHsnePlugin::onSelectionInColorMappingtSNERoi() {
    auto& lock = _selectionLocks[_colorImgRoitSNE->getId().toStdString()];
    lock++;
    if (lock.isLocked())
        return;

    Log::trace("onSelectionInColorMappingtSNERoi");

    // Selection in color map (overlayed on image) maps to image data
    // no call to selectionMapping() since we can just copy the same selection IDs
    const auto& sel = _colorImgRoitSNE->getSelection<Points>()->indices;
    auto inputData = getInputDataset<Points>();
    inputData->getSelection<Points>()->indices.assign(sel.cbegin(), sel.cend());
    events().notifyDatasetDataSelectionChanged(inputData);
}

void InteractiveHsnePlugin::deselectAll()
{
    Log::debug("InteractiveHsnePlugin::deselectAll");
    auto inputDataset = getInputDataset<Points>();

    inputDataset->getSelection<Points>()->indices.clear();
    events().notifyDatasetDataSelectionChanged(inputDataset);
}

void InteractiveHsnePlugin::continueComputation()
{
    Log::info("InteractiveHsnePlugin::continueComputation");
    TsneAnalysis* tsneAnalysis = &(_hsneSettingsAction->getInteractiveScaleAction().getTsneAnalysis());

    QString emdDataset = _hsneSettingsAction->getTsneSettingsAction().getGeneralTsneSettingsAction().getCurrentEmbDataset();

    if (emdDataset == _tSNEofROI->getId())
        tsneAnalysis = &_tsneROIAnalysis;

    if (emdDataset == _tSNEofLandmarks->getId())
        tsneAnalysis = &_tsneLandmarksAnalysis;

    if (tsneAnalysis->threadIsRunning())
    {
        uint32_t currentIterations = _hsneSettingsAction->getTsneSettingsAction().getGeneralTsneSettingsAction().getNumComputatedIterationsAction().getValue();
        uint32_t newIterations = _hsneSettingsAction->getTsneSettingsAction().getGeneralTsneSettingsAction().getNumNewIterationsAction().getValue();
        tsneAnalysis->continueComputation(currentIterations + newIterations);
    }

}

void InteractiveHsnePlugin::stopComputation()
{
    Log::info("InteractiveHsnePlugin::stopComputation");
    auto& tsneAnalysis = _hsneSettingsAction->getInteractiveScaleAction().getTsneAnalysis();

    if (tsneAnalysis.threadIsRunning())
        tsneAnalysis.stopComputation();

    if (_tsneROIAnalysis.threadIsRunning())
        _tsneROIAnalysis.stopComputation();

    if (_tsneLandmarksAnalysis.threadIsRunning())
        _tsneLandmarksAnalysis.stopComputation();

}

void InteractiveHsnePlugin::computeTSNEforROI()
{
    Log::debug("InteractiveHsnePlugin::computeTSNEforROI()");

    // get ID of ROI
    std::vector<uint32_t> imageSelectionIDs;
    utils::extractIdBlock(_layerRoiBottomLeft, _layerRoiTopRight, _hsneSettingsAction->getInteractiveScaleAction().getImageIndices(), imageSelectionIDs);
    assert(std::is_sorted(imageSelectionIDs.cbegin(), imageSelectionIDs.cend()));

    Log::info("InteractiveHsnePlugin: compute ROI t-SNE for " + std::to_string(imageSelectionIDs.size()) + " pixels");
    Log::debug(fmt::format("ROI: layerBottomLeft.x {0}, layerBottomLeft.y {1}, layerTopRight.x {2}, layerTopRight.y {3}", _layerRoiBottomLeft.x(), _layerRoiBottomLeft.y(), _layerRoiTopRight.x(), _layerRoiTopRight.y()));

    // Get number of enabled dimensions (c++ does not yet support capturing structured bindings in lambdas)
    std::vector<uint32_t> enabledDimensionsIDs;
    size_t numEnabledDimensions;
    std::tie(enabledDimensionsIDs, numEnabledDimensions) = enabledDimensions();

    // get data from ROI
    std::vector<float> dataROI;
    dataROI.resize(enabledDimensionsIDs.size() * imageSelectionIDs.size());
    auto inputData = getInputDataset<Points>();
    inputData->populateDataForDimensions<std::vector<float>, std::vector<uint32_t>, std::vector<uint32_t>>(dataROI, enabledDimensionsIDs, imageSelectionIDs);

    // prepare selection mapping
    _tSNEofROI->selectNone();
    events().notifyDatasetDataSelectionChanged(_tSNEofROI);
    _mappingROItSNEtoImage.clear();
    _mappingROItSNEtoImage.resize(imageSelectionIDs.size());
    _mappingImageToROItSNE.clear();
    _mappingImageToROItSNE.resize(inputData->getNumPoints());

    // reset lock to enable correct mapping
    _selectionLocks.visit([this](std::string name, utils::CyclicLock& lock) ->void {lock.reset(); });

    Log::trace("InteractiveHsnePlugin:: begin creating selection maps _mappingROItSNEtoImage and _mappingImageToROItSNE");

    // Create selection map
    uint32_t posInEmbedding = 0;
    for (const auto& imageSelectionID : imageSelectionIDs) {
        // add selection map entry
        _mappingROItSNEtoImage[posInEmbedding].push_back(imageSelectionID);
        _mappingImageToROItSNE[imageSelectionID].push_back(posInEmbedding);
        posInEmbedding++;
    }

    Log::trace("InteractiveHsnePlugin:: begin _tsneROIAnalysis");

    // Compute t-SNE of ROI
    _tsneROIAnalysis.stopComputation();
    TsneParameters tsneParameters = _hsneSettingsAction->getTsneSettingsAction().getTsneParameters();
    _tsneROIAnalysis.startComputation(tsneParameters, dataROI, static_cast<uint32_t>(numEnabledDimensions));

    _tSNEofROI->setProperty("Init", true);
}


void InteractiveHsnePlugin::computeTSNEforLandmarks()
{
    Log::debug("InteractiveHsnePlugin::computeTSNEforLandmarks()");
    TsneParameters tsneParameters = _hsneSettingsAction->getTsneSettingsAction().getTsneParameters();

    auto inputData = getInputDataset<Points>();

    // IDMap: Key -> Data ID, Value -> EmbIdAndPos: localIdOnScale, posInEmbedding
    const auto& idMap = _hsneSettingsAction->getInteractiveScaleAction().getIDMap();

    Log::info("InteractiveHsnePlugin: compute ROI t-SNE for " + std::to_string(idMap.size()) + " landmarks");

    // prepare selection mapping
    _tSNEofLandmarks->selectNone();
    events().notifyDatasetDataSelectionChanged(_tSNEofLandmarks);
    _mappingLandmarktSNEtoImage.clear();
    _mappingLandmarktSNEtoImage.resize(idMap.size());
    _mappingImageToLandmarktSNE.clear();
    _mappingImageToLandmarktSNE.resize(inputData->getNumPoints());

    // reset lock to enable correct mapping
    _selectionLocks.visit([this](std::string name, utils::CyclicLock& lock) ->void {lock.reset(); });

    Log::trace("InteractiveHsnePlugin:: begin creating selection maps _mappingROItSNEtoImage and _mappingImageToROItSNE");

    // create selecion maps and copy data
    std::vector<uint32_t> imageSelectionIDs;
    for (const auto& [dataID, embIdAndPos] : idMap) // Key -> Data ID, Value -> EmbIdAndPos: localIdOnScale, posInEmbedding
    {
        // add selection map entry
        _mappingLandmarktSNEtoImage[embIdAndPos.posInEmbedding].push_back(dataID);
        _mappingImageToLandmarktSNE[dataID].push_back(embIdAndPos.posInEmbedding);

        // selection IDs for data copying
        imageSelectionIDs.emplace_back(dataID);
    }
    std::sort(utils::exec_policy, imageSelectionIDs.begin(), imageSelectionIDs.end());

    // store data from of landmarks
    std::vector<float> dataLandmarks;
    size_t numEnabledDimensions;

    // Get number of enabled dimensions (c++ does not yet support capturing structured bindings in lambdas)
    std::vector<uint32_t> enabledDimensionsIDs;
    std::tie(enabledDimensionsIDs, numEnabledDimensions) = enabledDimensions();

    // copy landmark data from core data set
    dataLandmarks.resize(enabledDimensionsIDs.size() * imageSelectionIDs.size());
    inputData->populateDataForDimensions<std::vector<float>, std::vector<uint32_t>, std::vector<uint32_t>>(dataLandmarks, enabledDimensionsIDs, imageSelectionIDs);
    assert(dataLandmarks.size() == idMap.size() * numEnabledDimensions);

    Log::trace("InteractiveHsnePlugin:: begin _tsneLandmarksAnalysis");

    // Compute t-SNE of ROI
    _tsneLandmarksAnalysis.stopComputation();
    _tsneLandmarksAnalysis.startComputation(tsneParameters, dataLandmarks, static_cast<uint32_t>(numEnabledDimensions));

    _tSNEofLandmarks->setProperty("Init", true);
}

std::tuple<std::vector<uint32_t>, size_t> InteractiveHsnePlugin::enabledDimensions() const
{
    Log::trace("InteractiveHsnePlugin:: enabledDimensions");

    std::vector<bool> enabledDimensions = _hsneSettingsAction->getDimensionSelectionAction().getPickerAction().getEnabledDimensions();
    std::vector<uint32_t> enabledDimensionsIDs;
    for (uint32_t i = 0; i < _input[0].get<Points>()->getNumDimensions(); i++)
        if (enabledDimensions[i])
            enabledDimensionsIDs.push_back(i);

    const size_t numEnabledDimensions = count_if(enabledDimensions.begin(), enabledDimensions.end(), [](bool b) { return b; });

    return { enabledDimensionsIDs, numEnabledDimensions };
}

void InteractiveHsnePlugin::setColorMapData(Dataset<Points> emb, LandmarkMap& mapEmbToImg, Dataset<Points> imgDat, Dataset<Points> scatDat, const QImage& texture, std::vector<float>& imgColors, std::vector<float>& scatterColors)
{
    utils::ScopedTimer colorMapTimer("InteractiveHsnePlugin::setColorMapData", Log::debug);
    Log::debug(fmt::format("InteractiveHsnePlugin::setColorMapData: from embedding {0} to image data {1}", emb->getGuiName().toStdString(), imgDat->getGuiName().toStdString()));

    const size_t numImagePoints = static_cast<size_t>(_inputImageSize.height() * _inputImageSize.width());
    const uint32_t numEmbPoints = emb->getNumPoints();
    const size_t numColorChannels = 3;

    std::vector<float> embData(static_cast<size_t>(numEmbPoints) * 2u);
    std::vector<uint32_t> embDims{ 0, 1 };
    emb->populateDataForDimensions(embData, embDims);

    scatterColors.resize(static_cast<size_t>(numEmbPoints) * 3u);

    // Compute current embedding extends
    auto embeddingExtends = utils::computeExtends(embData);

    // Upscale texture, cause why not
    QImage textureScaled = texture.scaled(texture.size().width() * 2, texture.size().height() * 2, Qt::IgnoreAspectRatio, Qt::SmoothTransformation); // upscale with bilinear interpolation
    Log::trace(fmt::format("InteractiveHsnePlugin::setColorMapData: Texture size (orig) (w: {0}, h: {1})", texture.size().width(), texture.size().height()));
    Log::trace(fmt::format("InteractiveHsnePlugin::setColorMapData: Texture size (rescaled) (w: {0}, h: {1})", textureScaled.size().width(), textureScaled.size().height()));

    // Prepare lookup: map from embedding coordinates to texture extends
    const float x_range = embeddingExtends.extend_x();
    const float y_range = embeddingExtends.extend_y();
    const float x_min = embeddingExtends.x_min();
    const float y_min = embeddingExtends.y_min();
    const int tex_width = textureScaled.width() - 1;
    const int tex_height = textureScaled.height() - 1;

    auto map_x = [x_min, x_range, tex_width](float x) -> int32_t {
        return static_cast<int32_t>(tex_width * (x - x_min) / x_range);
    };

    auto map_y = [y_min, y_range, tex_height](float y) -> int32_t {
        return static_cast<int32_t>(tex_height * (y - y_min) / y_range);
    };

    // Lookup image color in texture based on embedding position
    const uint32_t greyVal = 128;
    const QRgb colorBg = QColor(greyVal, greyVal, greyVal).rgb();

    // fill with background color
    std::fill(utils::exec_policy, imgColors.begin(), imgColors.end(), greyVal);

    assert(mapEmbToImg.size() == numEmbPoints);

    // mapEmbToImg: embID to vec of ImgIDs
    QRgb color;
    for (size_t embID = 0; embID < numEmbPoints; embID++)
    {
        color = textureScaled.pixel(map_x(embData[embID * 2u]), map_y(embData[embID * 2u + 1u]));

        scatterColors[embID * numColorChannels] = qRed(color);
        scatterColors[embID * numColorChannels + 1u] = qGreen(color);
        scatterColors[embID * numColorChannels + 2u] = qBlue(color);

        // map the current color to all image points on which embID has the highest influence
        for (const auto& imgID : mapEmbToImg[embID])
        {
            imgColors[imgID * numColorChannels] = qRed(color);
            imgColors[imgID * numColorChannels + 1u] = qGreen(color);
            imgColors[imgID * numColorChannels + 2u] = qBlue(color);
        }

    }

    imgDat->setData(imgColors.data(), numImagePoints, numColorChannels);
    events().notifyDatasetDataChanged(imgDat);

    scatDat->setData(scatterColors.data(), numEmbPoints, numColorChannels);
    events().notifyDatasetDataChanged(scatDat);
}

void InteractiveHsnePlugin::setScatterColorMapData(Dataset<Points> emb, Dataset<Points> scatDat, const QImage& texture, std::vector<float>& scatterColors)
{
    utils::ScopedTimer colorMapTimer("InteractiveHsnePlugin::setScatterColorMapData", Log::debug);
    Log::debug(fmt::format("InteractiveHsnePlugin::setScatterColorMapData: for embedding {0}", emb->getGuiName().toStdString()));

    const uint32_t numEmbPoints = emb->getNumPoints();
    const size_t numColorChannels = 3;

    std::vector<float> embData(static_cast<size_t>(numEmbPoints) * 2u);
    std::vector<uint32_t> embDims{ 0, 1 };
    emb->populateDataForDimensions(embData, embDims);

    scatterColors.resize(static_cast<size_t>(numEmbPoints) * 3u);

    // Compute current embedding extends
    auto embeddingExtends = utils::computeExtends(embData);

    // Upscale texture, cause why not
    QImage textureScaled = texture.scaled(texture.size().width() * 2, texture.size().height() * 2, Qt::IgnoreAspectRatio, Qt::SmoothTransformation); // upscale with bilinear interpolation
    Log::trace(fmt::format("InteractiveHsnePlugin::setScatterColorMapData: Texture size (orig) (w: {0}, h: {1})", texture.size().width(), texture.size().height()));
    Log::trace(fmt::format("InteractiveHsnePlugin::setScatterColorMapData: Texture size (rescaled) (w: {0}, h: {1})", textureScaled.size().width(), textureScaled.size().height()));

    // Prepare lookup: map from embedding coordinates to texture extends
    const float x_range = embeddingExtends.extend_x();
    const float y_range = embeddingExtends.extend_y();
    const float x_min = embeddingExtends.x_min();
    const float y_min = embeddingExtends.y_min();
    const int tex_width = textureScaled.width() - 1;
    const int tex_height = textureScaled.height() - 1;

    auto map_x = [x_min, x_range, tex_width](float x) -> int32_t {
        return static_cast<int32_t>(tex_width * (x - x_min) / x_range);
    };

    auto map_y = [y_min, y_range, tex_height](float y) -> int32_t {
        return static_cast<int32_t>(tex_height * (y - y_min) / y_range);
    };

    // Lookup image color in texture based on embedding position
    const uint32_t greyVal = 128;
    const QRgb colorBg = QColor(greyVal, greyVal, greyVal).rgb();

    QRgb color;
    for (size_t embID = 0; embID < numEmbPoints; embID++)
    {
        color = textureScaled.pixel(map_x(embData[embID * 2u]), map_y(embData[embID * 2u + 1u]));

        scatterColors[embID * numColorChannels] = qRed(color);
        scatterColors[embID * numColorChannels + 1u] = qGreen(color);
        scatterColors[embID * numColorChannels + 2u] = qBlue(color);
    }

    scatDat->setData(scatterColors.data(), numEmbPoints, numColorChannels);
    events().notifyDatasetDataChanged(scatDat);
}


void InteractiveHsnePlugin::setColorBasedOnClusters() {
    utils::ScopedTimer colorMapTimer("InteractiveHsnePlugin::setColorBasedOnClusters", Log::debug);
    Log::debug(fmt::format("InteractiveHsnePlugin::setColorBasedOnClusters: from embedding {0} to image data {1}", _firstEmbedding->getGuiName().toStdString(), _colorImgTopLevelEmb->getGuiName().toStdString()));

    const size_t numImagePoints = static_cast<size_t>(_inputImageSize.height() * _inputImageSize.width());
    const uint32_t numEmbPoints = _firstEmbedding->getNumPoints();
    const size_t numColorChannels = 3;

    _scatterColorsTopLevelEmb.resize(static_cast<size_t>(numEmbPoints) * 3u);

    for (auto& cluster : _topLevelEmbClusters->getClusters()) {

        QRgb color = cluster.getColor().rgb();

        for (auto& embID : cluster.getIndices())
        {
            _scatterColorsTopLevelEmb[embID * numColorChannels] = qRed(color);
            _scatterColorsTopLevelEmb[embID * numColorChannels + 1u] = qGreen(color);
            _scatterColorsTopLevelEmb[embID * numColorChannels + 2u] = qBlue(color);

            // map the current color to all image points on which embID has the highest influence
            for (const auto& imgID : _topLevelEmbMapLocalToBottom[embID])
            {
                _imgColorsTopLevelEmb[imgID * numColorChannels] = qRed(color);
                _imgColorsTopLevelEmb[imgID * numColorChannels + 1u] = qGreen(color);
                _imgColorsTopLevelEmb[imgID * numColorChannels + 2u] = qBlue(color);
            }

        }

    }

    _colorImgTopLevelEmb->setData(_imgColorsTopLevelEmb.data(), numImagePoints, numColorChannels);
    events().notifyDatasetDataChanged(_colorImgTopLevelEmb);

    _colorScatterTopLevelEmb->setData(_scatterColorsTopLevelEmb.data(), numEmbPoints, numColorChannels);
    events().notifyDatasetDataChanged(_colorScatterTopLevelEmb);
}


void InteractiveHsnePlugin::setColorMapDataRoiHSNE() {
    std::vector<float> scatterColors;

    // Recolor image based on current embedding
    setColorMapData(getOutputDataset<Points>(), _mappingLocalToBottom, _colorImgRoiHSNE, _colorScatterRoiHSNE, _hsneSettingsAction->getInteractiveScaleAction().getColorMapRoiEmbAction().getColorMapImage(), _imgColorsRoiHSNE, scatterColors);
}

void InteractiveHsnePlugin::setScatterColorBasedOnTopLevel() {

    // Get scale informations
    const uint32_t currentScale = _hsneSettingsAction->getInteractiveScaleAction().getScaleLevel();
    const uint32_t topScale = _hierarchy.getTopScale();
    const auto& idMap = _hsneSettingsAction->getInteractiveScaleAction().getIDMap();
    const auto numEmbPoints = idMap.size();
    const size_t numImagePoints = static_cast<size_t>(_inputImageSize.height() * _inputImageSize.width());
    const size_t numColorChannels = 3;
    const uint32_t blackVal = 0;

    // Get landmark ID on scale
    std::vector<uint32_t> landmarkIDsOnScale(numEmbPoints);
    for (const auto& [dataID, embIdAndPosInScale] : idMap)
        landmarkIDsOnScale[embIdAndPosInScale.posInEmbedding] = embIdAndPosInScale.localIdOnScale;

    std::vector<float> imgColorsRoiHSNEBasedOnTopLevel(numImagePoints * 3u, blackVal);

    // map colors
    std::vector<float> scatterColors(numEmbPoints * numColorChannels);
    for (uint32_t localID = 0; localID < numEmbPoints; localID++)
    {
        // get representative ID on top scale for each scale embedding ID
        std::vector<uint32_t> imageSelectionID = { _hierarchy.getScale(currentScale)._landmark_to_original_data_idx[landmarkIDsOnScale[localID]] };
        std::vector<uint32_t> localIDsOnTopScale;
        utils::computeLocalIDsOnCoarserScaleHeuristic(topScale, imageSelectionID, _hierarchy, localIDsOnTopScale);
        //utils::computeLocalIDsOnCoarserScale(currentScale, imageSelectionIDs, hsneHierarchy, tresh_influence, localIDsOnCoarserScale)

        assert(localIDsOnTopScale.size() <= 1);

        if (localIDsOnTopScale.size() == 0)
        {
            Log::warn(fmt::format("InteractiveHsnePlugin::setColorMapDataRoiHSNE: embedding landmark {0} did not have a representative top level landmark", localID));

            scatterColors[localID * numColorChannels] = blackVal;
            scatterColors[localID * numColorChannels + 1u] = blackVal;
            scatterColors[localID * numColorChannels + 2u] = blackVal;
        }
        else
        {
            scatterColors[localID * numColorChannels] = _scatterColorsTopLevelEmb[localIDsOnTopScale[0] * numColorChannels];
            scatterColors[localID * numColorChannels + 1u] = _scatterColorsTopLevelEmb[localIDsOnTopScale[0] * numColorChannels + 1u];
            scatterColors[localID * numColorChannels + 2u] = _scatterColorsTopLevelEmb[localIDsOnTopScale[0] * numColorChannels + 2u];
        }

        for (const auto& imgID : _mappingLocalToBottom[localID])
        {
            imgColorsRoiHSNEBasedOnTopLevel[imgID * numColorChannels] = scatterColors[localID * 3ull];
            imgColorsRoiHSNEBasedOnTopLevel[imgID * numColorChannels + 1u] = scatterColors[localID * 3ull + 1u];
            imgColorsRoiHSNEBasedOnTopLevel[imgID * numColorChannels + 2u] = scatterColors[localID * 3ull + 2u];
        }

    }

    _colorImgRoiHSNEBasedOnTopLevel->setData(imgColorsRoiHSNEBasedOnTopLevel.data(), numImagePoints, numColorChannels);
    events().notifyDatasetDataChanged(_colorImgRoiHSNEBasedOnTopLevel);

    _colorEmbScatBasedOnTopLevelEmb->setData(scatterColors.data(), numEmbPoints, numColorChannels);
    events().notifyDatasetDataChanged(_colorEmbScatBasedOnTopLevelEmb);
}

void InteractiveHsnePlugin::setColorMapDataRoitSNE() {
    if (_tSNEofROI->getProperty("Init").toBool())
    {
        std::vector<float> scatterColors;
        setColorMapData(_tSNEofROI, _mappingROItSNEtoImage, _colorImgRoitSNE, _colorScatterRoitSNE, _hsneSettingsAction->getInteractiveScaleAction().getColorMapRoiEmbAction().getColorMapImage(), _imgColorstSNE, scatterColors);
    }
}

void InteractiveHsnePlugin::setColorMapDataTopLevelEmb() {
    if (!_firstEmbedding->getProperty("Init").toBool())
        return;
    
    if(!_hsneSettingsAction->getMeanShiftActionAction().getUseClusterColorsAction().isChecked())
    {
        // set colors of ROI embedding and recolored image based on 2D
        // sets _colorImgTopLevelEmb, _colorScatterTopLevelEmb, _imgColorsTopLevelEmb, _scatterColorsTopLevelEmb
        setColorMapData(_firstEmbedding, _topLevelEmbMapLocalToBottom, _colorImgTopLevelEmb, _colorScatterTopLevelEmb, _hsneSettingsAction->getInteractiveScaleAction().getColorMapFirstEmbAction().getColorMapImage(), _imgColorsTopLevelEmb, _scatterColorsTopLevelEmb);

        // sets _colorImgRoiHSNE, _colorScatterRoiHSNE, _imgColorsRoiHSNE
        setColorMapDataRoiHSNE();
    }
    else
    {
        // sets _colorImgTopLevelEmb, _colorScatterTopLevelEmb, _imgColorsTopLevelEmb, _scatterColorsTopLevelEmb, _colorImgRoiHSNE, _imgColorsRoiHSNE
        setColorBasedOnClusters();
    }

    // sets _colorImgRoiHSNEBasedOnTopLevel, _colorEmbScatBasedOnTopLevelEmb based on _scatterColorsTopLevelEmb
    setScatterColorBasedOnTopLevel();

}

void InteractiveHsnePlugin::saveCurrentColorImageAsPrev()
{
    _colorImgRoiHSNEprev->setData(_imgColorsRoiHSNE.data(), _colorImgRoiHSNEprev->getNumPoints(), _colorImgRoiHSNEprev->getNumDimensions());
    events().notifyDatasetDataChanged(_colorImgRoiHSNEprev);
}

uint32_t InteractiveHsnePlugin::compNumHierarchyScales()
{
    if (_hsneSettingsAction == nullptr || !_hsneSettingsAction->getAdvancedHsneSettingsAction().getHardCutOffAction().isChecked())
        return compNumHierarchyScalesLog();

    return compNumHierarchyScalesTarget(_hsneSettingsAction->getInteractiveScaleAction().getVisBudgetTargetSlider().getValue(),
                                        _hsneSettingsAction->getAdvancedHsneSettingsAction().getHardCutOffPercentageAction().getValue());

}

uint32_t InteractiveHsnePlugin::compNumHierarchyScalesLog()
{
    return std::max(1L, std::lround(log10(getInputDataset<Points>()->getNumPoints())) - 2);
}

uint32_t InteractiveHsnePlugin::compNumHierarchyScalesTarget(uint32_t target, float hardcutoff)
{
    float numPointsOnNewScale = static_cast<float>(getInputDataset<Points>()->getNumPoints());
    uint32_t numScales;

    for (numScales = 1; numPointsOnNewScale > target ; ++numScales)
    {
        numPointsOnNewScale *= hardcutoff;
    }

    return numScales;
}


/// //////////////////////////////////// ///
/// InteractiveHsneAnalysisPluginFactory ///
/// //////////////////////////////////// ///
InteractiveHsneAnalysisPluginFactory::InteractiveHsneAnalysisPluginFactory()
{
    const auto margin = 3;
    const auto pixmapSize = QSize(100, 100);
    const auto pixmapRect = QRect(QPoint(), pixmapSize).marginsRemoved(QMargins(margin, margin, margin, margin));
    const auto halfSize = pixmapRect.size() / 2;
    const auto thirdSize = pixmapRect.size() / 3;
    const auto quarterSize = pixmapRect.size() / 4;

    // Create pixmap
    QPixmap pixmap(pixmapSize);

    // Fill with a transparent background
    pixmap.fill(Qt::transparent);

    // Create a painter to draw in the pixmap
    QPainter painter(&pixmap);

    // Enable anti-aliasing
    painter.setRenderHint(QPainter::Antialiasing);

    // Configure painter
    painter.setPen(QPen(Qt::black, 1, Qt::SolidLine, Qt::SquareCap, Qt::SvgMiterJoin));
    painter.setFont(QFont("Arial", 32, 350));

    const auto textOption = QTextOption(Qt::AlignCenter);

    // Do the painting
    painter.drawText(QRect(pixmapRect.topLeft(), quarterSize), "I", textOption);
    painter.drawText(QRect(QPoint(quarterSize.width(), pixmapRect.top()), thirdSize), "M", textOption);
    painter.drawText(QRect(QPoint(thirdSize.width() * 2, pixmapRect.top()), thirdSize), "G", textOption);

    painter.setFont(QFont("Arial", 28, 200));
    painter.drawText(QRect(QPoint(pixmapRect.left(), halfSize.height()), quarterSize), "H", textOption);
    painter.drawText(QRect(QPoint(quarterSize.width(), halfSize.height()), quarterSize), "S", textOption);
    painter.drawText(QRect(QPoint(quarterSize.width() * 2, halfSize.height()), quarterSize), "N", textOption);
    painter.drawText(QRect(QPoint(quarterSize.width() * 3, halfSize.height()), quarterSize), "E", textOption);

    setIcon(QIcon(pixmap));
}

AnalysisPlugin* InteractiveHsneAnalysisPluginFactory::produce()
{
    return new InteractiveHsnePlugin(this);
}

mv::DataTypes InteractiveHsneAnalysisPluginFactory::supportedDataTypes() const
{
    DataTypes supportedTypes;
    supportedTypes.append(PointType);

    return supportedTypes;
}

PluginTriggerActions InteractiveHsneAnalysisPluginFactory::getPluginTriggerActions(const mv::Datasets& datasets) const
{
    PluginTriggerActions pluginTriggerActions;

    const auto getPluginInstance = [this](const Dataset<Points>& dataset) -> InteractiveHsnePlugin* {
        return dynamic_cast<InteractiveHsnePlugin*>(plugins().requestPlugin(getKind(), { dataset }));
    };

    if (PluginFactory::areAllDatasetsOfTheSameType(datasets, PointType)) {
        if (datasets.count() >= 1) {
            auto pluginTriggerAction = new PluginTriggerAction(const_cast<InteractiveHsneAnalysisPluginFactory*>(this), this, "IMG HSNE analysis", "Perform image HSNE analysis", icon(), [this, getPluginInstance, datasets](PluginTriggerAction& pluginTriggerAction) -> void {
                for (const auto& dataset : datasets)
                    getPluginInstance(dataset);
                });

            pluginTriggerActions << pluginTriggerAction;
        }
    }

    return pluginTriggerActions;
}

