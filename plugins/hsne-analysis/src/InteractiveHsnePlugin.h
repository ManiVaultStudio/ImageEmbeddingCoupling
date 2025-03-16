#pragma once

#include <AnalysisPlugin.h>

#include "HsneHierarchy.h"
#include "HsneSettingsAction.h"
#include "CommonTypes.h"

#include <QVector3D>
#include <QSize>

#include <chrono>
#include <unordered_map>
#include <tuple>
#include <memory>

using namespace mv::plugin;

class HsneScaleAction;
class TsneAnalysis;
class Clusters;
using LockSet = utils::Locks<utils::CyclicLock>;

namespace math {
    enum class PCA_ALG;
}

/**
 * InteractiveHsnePlugin
 *
 * Main plugin class responsible for computing interactive HSNE hierarchies over point data
 *
 * @author Alexander Vieth, Julian Thijssen
 */
class InteractiveHsnePlugin : public AnalysisPlugin
{
    Q_OBJECT
public:
    InteractiveHsnePlugin(const PluginFactory* factory);
    ~InteractiveHsnePlugin() override;

    void init() override;

    HsneHierarchy& getHierarchy() { return _hierarchy; }

    HsneSettingsAction& getHsneSettingsAction() { return *_hsneSettingsAction; }

    Dataset<Points> getPointInitTypesDataset() { return _pointInitTypes; }
    Dataset<Points> getRoiRepresentationDataset() { return _roiRepresentation; }
    Dataset<Points> getNumberTransitionsDataset() { return _numberTransitions; }
    Dataset<Points> getFirstEmbeddingDataset() { return _firstEmbedding; }
    Dataset<Points> getRegHsneTopLevelDataset() { return _regHsneTopLevel; }
    Dataset<Points> getTopLevelLandmarkDataDataset() { return _topLevelLandmarkData; }
    Dataset<Points> getRoiEmbLandmarkDataDataset() { return _roiEmbLandmarkData; }
    Dataset<Points> getSelectionAttributeDataDataset() { return _selectionAttributeData; }
    Dataset<Points> getColorScatterTopLevelEmbDataset() { return _colorScatterTopLevelEmb; }
    Dataset<Points> getColorEmbScatBasedonTopLevelEmbDataset() { return _colorEmbScatBasedOnTopLevelEmb; }
    Dataset<Points> getColorImgTopLevelEmbDataset() { return _colorImgTopLevelEmb; }
    Dataset<Points> getColorScatterRoiHSNEDataset() { return _colorScatterRoiHSNE; }
    Dataset<Points> getColorMappingDataset() { return _colorImgRoiHSNE; }
    Dataset<Points> getPrevColorMappingDataset() { return _colorImgRoiHSNEprev; }
    Dataset<Clusters> getTopLevelEmbClustersDataset() { return _topLevelEmbClusters; }

    bool isInitialized() const { return _initialized; }

    uint32_t compNumHierarchyScales();
    uint32_t compNumHierarchyScalesLog();
    uint32_t compNumHierarchyScalesTarget(uint32_t target, float hardcutoff);

public:
    void setSelectionMapLocalToBottom(LandmarkMap map) { _mappingLocalToBottom = map; }
    void setSelectionMapBottomToLocal(LandmarkMapSingle map) { _mappingBottomToLocal = map; }

    LandmarkMap& getSelectionMapLocalToBottom() { return _mappingLocalToBottom; }
    LandmarkMapSingle& getSelectionMapBottomToLocal() { return _mappingBottomToLocal; }
    LandmarkMap& getSelectionMapTopLevelEmbLocalToBottom() { return _topLevelEmbMapLocalToBottom; }
    LandmarkMapSingle& getSelectionMapTopLevelEmbBottomToLocal() { return _topLevelEmbMapBottomToLocal; }
    LandmarkMap& getSelectionMapTopLevelDataLocalToBottom() { return _topLevelDataMapLocalToBottom; }
    LandmarkMapSingle& getSelectionMapTopLevelDataBottomToLocal() { return _topLevelDataMapBottomToLocal; }
    LandmarkMap& getSelectionMapCurrentLevelDataLocalToBottom() { return _currentLevelDataMapLocalToBottom; }
    LandmarkMapSingle& getSelectionMapCurrentLevelDataBottomToLocal() { return _currentLevelDataMapBottomToLocal; }
    LandmarkMap& getSelectionMapSelectionDataLocalToBottom() { return _selectionAttributeDataMapLocalToBottom; }
    LandmarkMapSingle& getSelectionMapSelectionDataBottomToLocal() { return _selectionAttributeDataMapBottomToLocal; }

    void deselectAll();

    /** imgColors are not resized, scatterColors are resized*/
    void setColorMapData(Dataset<Points> emb, LandmarkMap& mapEmbToImg, Dataset<Points> imgDat, Dataset<Points> scatDat, const QImage& texture, std::vector<float>& imgColors, std::vector<float>& scatterColors);
    
    /** imgColors are not resized, scatterColors are resized*/
    void setScatterColorMapData(Dataset<Points> emb, Dataset<Points> scatDat, const QImage& texture, std::vector<float>& scatterColors);

private:
    /** The modified image viewer reports the current viewport 
    * A ViewportSharingActions is added to the input image, the modified image viewer recognizes this and emits ViewportSharingActions::viewportInImageChanged
    * which in turn triggers this slot.
    * The image viewer coord system y axis is flipped. Here we flip it into user coordinates: 
    *       roiTopLeft      (image viewer) becomes _layerRoiBottomLeft (here)
    *       roiBottomRight  (image viewer) becomes _layerRoiTopRight   (here)
    * 
    * layer roi -> image coordinates, view roi -> view coordinates (depends on viewer window size)
    */
    void updateImageViewport(const QVector3D layerRoiBottomLeft, const QVector3D layerRoiTopRight, const QVector3D viewRoiXY, const QVector3D viewRoiWH);

    /** A selection in the image is mapped to a selection in the embeddings using _mappingBottomToLocal, _mappingImageToROItSNE and _mappingImageToLandmarktSNE*/
    void onSelectionInImage();

    /** A selection in the embedding is mapped to a selection in the image using _mappingLocalToBottom */
    void onSelectionInEmbedding();

    /** A selection in the ROI t-SNE embedding is mapped to a selection in the image using _mappingROItSNEtoImage */
    void onSelectionInROItSNE();

    /** A selection in the landmark t-SNE embedding is mapped to a selection in the image using _mappingLandmarktSNEtoImage */
    void onSelectionInLandmarktSNE();

    /** A selection in the color ROI HSNE is mapped to a selection in the image */
    void onSelectionInColorMappingHsneRoi(Dataset<Points> colorImgRoiHSNE);

    /** A selection in the color ROI tSNE is mapped to a selection in the image */
    void onSelectionInColorMappingtSNERoi();

    /** A selection in the first top scale embedding */
    void onSelectionFirstEmbedding();

    /** A selection in the first top scale landmark data */
    void onSelectionFirstEmbeddingData();

    /** A selection in the first top scale landmark data */
    void onSelectionCurrentLevelLandmarkData();

    /** A selection in the selection attribute data */
    void onSelectionSelectionLandmarkData();

    /** A selection in the top level embedding recolored image */
    void onSelectionTopLevelImage();

    /** A selection in the top level embedding for regular HSNE */
    void onSelectionRegHsneTopLevelEmbedding();

    /** Maps a selection from one data set to another (either embedding to image or vice versa) using a selection mapping */
    void selectionMapping(const mv::Dataset<Points> selectionInputData, const LandmarkMap& selectionMap, mv::Dataset<Points> selectionOutputData, utils::CyclicLock& lock);
    
    void selectionMapping(const mv::Dataset<Points> selectionInputData, const LandmarkMapSingle& selectionMap, mv::Dataset<Points> selectionOutputData, utils::CyclicLock& lock);

    void continueComputation();
    void stopComputation();

    void computeTSNEforROI();

    void computeTSNEforLandmarks();

    std::tuple<std::vector<uint32_t>, size_t> enabledDimensions() const;

protected:
    void setColorMapDataRoiHSNE();
    void setColorMapDataRoitSNE();
    void setColorMapDataTopLevelEmb();

    void setColorBasedOnClusters();

    // Recolor current embedding based on top level embedding
    void setScatterColorBasedOnTopLevel();

    void saveCurrentColorImageAsPrev();

private:
    std::shared_ptr<HsneSettingsAction>     _hsneSettingsAction;        /** Pointer to HSNE settings action */
    HsneHierarchy         _hierarchy;                   /** HSNE hierarchy */

    LandmarkMap           _mappingLocalToBottom;        /** Maps embedding indices to bottom indices (in image). The embedding indices refer to their position in the dataset vector */
    LandmarkMapSingle     _mappingBottomToLocal;        /** Maps bottom indices (in image) to embedding indices. The embedding indices refer to their position in the dataset vector */
    LandmarkMap           _topLevelEmbMapLocalToBottom; /**  */
    LandmarkMapSingle     _topLevelEmbMapBottomToLocal; /**  */
    LandmarkMap           _topLevelDataMapLocalToBottom;/**  */
    LandmarkMapSingle     _topLevelDataMapBottomToLocal;/**  */
    LandmarkMap           _currentLevelDataMapLocalToBottom;/**  */
    LandmarkMapSingle     _currentLevelDataMapBottomToLocal;/**  */
    LandmarkMap           _selectionAttributeDataMapLocalToBottom;/**  */
    LandmarkMapSingle     _selectionAttributeDataMapBottomToLocal;/**  */

    LockSet                 _selectionLocks;            /** Prevents endless selection loop */

    QSize                   _inputImageSize;            /** Size (width and height) of the input dataset image */
    std::string             _inputImageLoadPath;        /** Image load path */
    utils::Vector2D         _layerRoiBottomLeft;        /** ROI bottom left, .x() is width and .y() is height. (0,0) is buttom left from user perspective, x-axis goes to the right */
    utils::Vector2D         _layerRoiTopRight;          /** ROI top right left, .x() is width and .y() is height. (0,0) is buttom left from user perspective, x-axis goes to the right */
    utils::Vector2D         _viewRoiXY;         /** ROI bottom left, .x() is width and .y() is height. (0,0) is buttom left from user perspective, x-axis goes to the right */
    utils::Vector2D         _viewRoiWH;           /** ROI top right left, .x() is width and .y() is height. (0,0) is buttom left from user perspective, x-axis goes to the right */

    bool                    _initialized;               /** Set to true after first embedding is computed in init() */

    Dataset<Points>         _pointInitTypes;            /** PointInitTypes dataset reference */
    Dataset<Points>         _roiRepresentation;         /** Roi representation dataset reference */
    Dataset<Points>         _numberTransitions;         /** Transitions number dataset reference */
    Dataset<Points>         _colorScatterRoiHSNE;       /** Color scatter dataset reference */
    Dataset<Points>         _colorImgRoiHSNE;           /** Color image dataset reference */
    Dataset<Points>         _colorImgRoiHSNEBasedOnTopLevel; /** Color image dataset based on top level reference */
    Dataset<Points>         _colorImgRoiHSNEprev;       /** Previous Color image dataset reference */
    Dataset<Points>         _firstEmbedding;            /** Top scale level embedding dataset reference */
    Dataset<Points>         _regHsneTopLevel;           /** Top scale level embedding for HSNE dataset reference */
    Dataset<Points>         _topLevelLandmarkData;      /** Top scale level landmarks data dataset reference */
    Dataset<Points>         _colorImgTopLevelEmb;       /** Color image of first top scale level embedding*/
    Dataset<Points>         _colorScatterTopLevelEmb;   /** Color scatter of first top scale level embedding*/
    Dataset<Points>         _colorEmbScatBasedOnTopLevelEmb;   /** Recoloring of current embedding based on the recoloring of the top-level embedding */
    Dataset<Points>         _roiEmbLandmarkData;  /** Top scale level landmarks data dataset reference */
    Dataset<Points>         _selectionAttributeData;    /** Selection data dataset reference */
    Dataset<Clusters>       _topLevelEmbClusters;       /** Selection data dataset reference */

    Dataset<Points>         _tSNEofROI;                 /** t-SNE of ROI dataset reference */
    Dataset<Points>         _colorImgRoitSNE;           /** Color image dataset reference */
    Dataset<Points>         _colorScatterRoitSNE;       /** Color scatter dataset reference */
    TsneAnalysis            _tsneROIAnalysis;           /** TSNE ROI analysis */
    LandmarkMap             _mappingROItSNEtoImage;     /** Maps ROI t-SNE indices to image indices. */
    LandmarkMap             _mappingImageToROItSNE;     /** Maps image indices to ROI t-SNE indices. */

    Dataset<Points>         _tSNEofLandmarks;           /** t-SNE of landmarks */
    TsneAnalysis            _tsneLandmarksAnalysis;     /** TSNE Landmarks analysis */
    LandmarkMap             _mappingLandmarktSNEtoImage;/** Maps landmark t-SNE indices to image indices. */
    LandmarkMap             _mappingImageToLandmarktSNE;/** Maps image indices to landmark t-SNE indices. */

    std::vector<float>      _imgColorsRoiHSNE;          /** used to save the previous recoloring */
    std::vector<float>      _imgColorstSNE;
    std::vector<float>      _imgColorsTopLevelEmb;
    std::vector<float>      _scatterColorsTopLevelEmb;

    friend                  HsneScaleAction;            /** Easy access to private functions */
};

class InteractiveHsneAnalysisPluginFactory : public AnalysisPluginFactory
{
    Q_INTERFACES(mv::plugin::AnalysisPluginFactory mv::plugin::PluginFactory)
    Q_OBJECT
    Q_PLUGIN_METADATA(IID   "nl.tudelft.InteractiveHsnePlugin"
                      FILE  "InteractiveHsnePlugin.json")

public:
    InteractiveHsneAnalysisPluginFactory();
    ~InteractiveHsneAnalysisPluginFactory() override {}

    AnalysisPlugin* produce() override;

    /** Returns the data types that are supported by the example analysis plugin */
    mv::DataTypes supportedDataTypes() const override;

    /**
     * Get plugin trigger actions given \p datasets
     * @param datasets Vector of input datasets
     * @return Vector of plugin trigger actions
     */
    PluginTriggerActions getPluginTriggerActions(const mv::Datasets& datasets) const override;
};
