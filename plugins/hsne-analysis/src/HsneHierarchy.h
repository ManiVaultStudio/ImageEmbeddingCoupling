#pragma once

#include "CoreInterface.h"
#include "CommonTypes.h"
#include "Logger.h"

#include "hdi/utils/graph_algorithms.h"
#include "hdi/utils/cout_log.h"

#include <QString>

#include <vector>
#include <unordered_map>
#include <memory>
#include <string>
#include <filesystem>

class Points;
class HsneParameters;
class HsneHierarchy;

using Path = std::filesystem::path;

/**
 * InfluenceHierarchy
 *
 * Container class for the mapping of the HSNE scales to the data level
 *
 * @author Julian Thijssen
 */
class InfluenceHierarchy
{
public:
    void initialize(const HsneHierarchy& hierarchy);

    std::vector<LandmarkMap>& getMapTopDown() { return _influenceMapTopDown; }
    const std::vector<LandmarkMap>& getMapTopDown() const { return _influenceMapTopDown; }

    std::vector<LandmarkMap>& getMapBottomUp() { return _influenceMapBottomUp; }
    const std::vector<LandmarkMap>& getMapBottomUp() const { return _influenceMapBottomUp; }

private:
    /** Size: number of scales.
    * For each scale a Landmarkmap: each Landmarkmap is of the size of landmarks on a scale
    * landmarkMap[i] is a vector of data points (global IDs)  on which the landmark i has the highest influence
    * _influenceMapTopDown[scale][LandmarkIDOnScale] -> vector of data point IDs for which LandmarkIDOnScale has the highest influence on
    */
    std::vector<LandmarkMap> _influenceMapTopDown;

    /** Reverse mapping of _influenceMapTopDown
    * Size: number of scales.
    * For each scale a Landmarkmap: each Landmarkmap is of the size data points
    * landmarkMap[i] is a vector of landmarks IDs (on scale) which influence data point i the most, this is either 0 or 1 landmark
    * _influenceMapBottomUp[scale][dataPointID] -> vector of (scale-relative) landmarks that influence dataPointID
    */
    std::vector<LandmarkMap> _influenceMapBottomUp;
};

/**
 * HsneHierarchy
 *
 * Wrapper for the HDI HSNE hierarchy
 *
 * @author Julian Thijssen
 */
class HsneHierarchy : public QObject
{
    Q_OBJECT
public:
    /**
     * Initialize the HSNE hierarchy with a data-level scale.
     *
     * @param  data        The high-dimensional data
     * @param  parameters  Parameters with which to run the HSNE algorithm
     */
    void initialize(mv::CoreInterface* core, const Points& inputData, const std::vector<bool>& enabledDimensions, const HsneParameters& parameters, const std::string& cachePath = std::string());
    HsneMatrix getTransitionMatrixAtScale(uint32_t scale) { return _hsne->scale(scale)._transition_matrix; }
    const HsneMatrix& getTransitionMatrixAtScale(uint32_t scale) const { return _hsne->scale(scale)._transition_matrix; }

    void printScaleInfo()
    {
        Log::info("Landmark to Orig size: " + std::to_string(_hsne->scale(getNumScales() - 1)._landmark_to_original_data_idx.size()));
        Log::info("Landmark to Prev size: " + std::to_string(_hsne->scale(getNumScales() - 1)._landmark_to_previous_scale_idx.size()));
        Log::info("Prev to Landmark size: " + std::to_string(_hsne->scale(getNumScales() - 1)._previous_scale_to_landmark_idx.size()));
        Log::info("AoI size: " + std::to_string(_hsne->scale(getNumScales() - 1)._area_of_influence.size()));
    }

    const Hsne::scale_type& getScale(uint32_t scaleId) const
    {
        return _hsne->scale(scaleId);
    }

    Hsne::scale_type& getScale(uint32_t scaleId)
    {
        return _hsne->scale(scaleId);
    }

    const InfluenceHierarchy& getInfluenceHierarchy() const
    {
        return _influenceHierarchy;
    }

    InfluenceHierarchy& getInfluenceHierarchy()
    {
        return _influenceHierarchy;
    }

    const std::vector<std::vector<uint32_t>>& getTransitionNNOnScale(uint32_t scale) const
    {
        return _transitionNNOnScale[scale];
    }

    /**
     * Returns a map of landmark indices and influences on the refined scale (currentScale - 1) in the hierarchy,
     * that are influenced by landmarks specified by their index in the current scale.
     * 
     * Sets neighbors parameter.
     */
    void getInfluencedLandmarksInRefinedScale(uint32_t currentScale, const std::vector<uint32_t>& indices, std::map<uint32_t, float>& neighbors) const
    {
        _hsne->getInfluencedLandmarksInPreviousScale(currentScale, const_cast<std::vector<uint32_t>&>(indices), neighbors);
    }

    /**
     * Returns a map of landmark indices and influences on the coarser scale (currentScale + 1) in the hierarchy,
     * that influence the specified landmarks given by their index in the current scale.
     *
     * Sets neighbors.
     */
    void getInfluencingLandmarksInCoarserScale(const uint32_t currentScale, const std::vector<uint32_t>& indices, std::map<uint32_t, float>& neighbors) const
    {
        _hsne->getInfluencingLandmarksInNextScale(currentScale, const_cast<std::vector<uint32_t>&>(indices), neighbors);    // Note: As of April 2022, the docs mention currentScale + 1 but that's a mistake
    }

    /**
     * Return the influence exercised on the data point dataPointId by the landmarks in each scale
     * 
     * Sets influence.
     */
    void getInfluenceOnDataPoint(const uint32_t dataPointId, std::vector<std::unordered_map<uint32_t, float>>& influence, float thresh = 0, bool normalized = true) const
    {
        _hsne->getInfluenceOnDataPoint(dataPointId, influence, thresh, normalized);
    }

    /**
     * Extract a part of the transition matrix at a given scale for the specified landmarkIdxs on that scale
     */
    void getTransitionMatrixForSelectionAtScale(const uint32_t scale, const uint32_t threshConnections, std::vector<uint32_t>& landmarkIdxs, HsneMatrix& transitionMatrix, float thresh = 0.0f) const;
    
    /**
    * Extract a part of the transition matrix at a given scale for the specified landmarkIdxs on that scale
    */
    void getTransitionMatrixForSelectionAtScale(const uint32_t scale, std::vector<uint32_t>& landmarkIdxs, HsneMatrix& transitionMatrix) const;
    
    /**
     * Returns local IDs of landmarks at coarser currentScale+1 that are influencING landmarkIdxs at currentScale (above a treshold)
     */
    void getLocalIDsInCoarserScale(uint32_t currentScale, const std::vector<uint32_t>& landmarkIdxs, std::vector<uint32_t>& coarserScaleIDxs, float tresh = 0.5) const;

    /**
     * Returns local IDs of landmarks at refined currentScale-1 that are influencED BY landmarkIdxs at currentScale (above a treshold)
     */
    void getLocalIDsInRefinedScale(uint32_t currentScale, const std::vector<uint32_t>& landmarkIdxs, std::vector<uint32_t>& refinedScaleIDxs, float tresh = 0.5) const;

    /**
     * Compute maps between embedding IDs and bottom IDs (in image) used for interactive selection
     * localIDsOnScale refers to the local ID wrt the scale not the embedding (since the embedding might contain a subset of IDs of a scale)
     */
    void computeSelectionMapsAtScale(const uint32_t scale, const std::vector<uint32_t>& localIDsOnNewScale, LandmarkMapSingle& mappingBottomToLocal, LandmarkMap& mappingLocalToBottom) const;

    uint32_t getNumScales() const { return _numScales; }
    uint32_t getTopScale() const { return _numScales - 1; }
    QString getInputDataName() const { return _inputDataName; }
    uint32_t getNumPoints() const { return _numPoints; }
    uint32_t getNumDimensions() const { return _numDimensions; }

    /** Save HSNE hierarchy from this class to disk */
    void saveCacheHsne() const;

    /** Load HSNE hierarchy from disk */
    bool loadCache();

private:
    /** Save HsneHierarchy to disk */
    void saveCacheHsneHierarchy(std::string fileName) const;
    /** Save InfluenceHierarchy to disk */
    void saveCacheHsneInfluenceHierarchy(std::string fileName, const std::vector<LandmarkMap>& influenceHierarchy) const;
    /** Save InfluenceHierarchy to disk */
    void saveCacheHsneTransitionNNOnScale(std::string fileName) const;
    /** Save HSNE parameters to disk */
    void saveCacheParameters(std::string fileName) const;

    /** Check whether HSNE parameters of the cached values on disk correspond with the current settings */
    bool checkCacheParameters(std::string fileName) const;
    /** Load HsneHierarchy from disk */
    bool loadCacheHsneHierarchy(std::string fileName);
    /** Load InfluenceHierarchy from disk */
    bool loadCacheHsneInfluenceHierarchy(std::string fileName, std::vector<LandmarkMap>& influenceHierarchy);
    /** Load InfluenceHierarchy from disk */
    bool loadCacheHsneTransitionNNOnScale(std::string fileName);

    /** Sets the hsne parameter member variables */
    void setParameters(const HsneParameters& params);

    /** Compute the nearest neighbor for each landmark on each scale based on the transition matrices */
    void computeTransitionNN();

    void computeSimilarities(const std::vector<float>& data);

private:
    mv::CoreInterface* _core;                 /**  */

    Hsne::Parameters _params;                   /**  */
    bool _exactKnn;                             /** Compute Exact KNN instead of approximation */

    std::unique_ptr<Hsne> _hsne;                /**  */
    InfluenceHierarchy _influenceHierarchy;     /**  */

    //   scales     landmarks      kNN      ID 
    std::vector<std::vector<std::vector<uint32_t>>> _transitionNNOnScale;   /**  */

    std::unique_ptr<hdi::utils::CoutLog> _log;  /**  */

    QString _inputDataName;                     /**  */

    uint32_t _numScales;                    /**  */

    uint32_t _numPoints;                    /**  */
    uint32_t _numDimensions;                /**  */

    HsneMatrix _similarities;                   /** only populated if exact knn are asked for */

    Path _cachePath;                            /** Path for saving and loading cache */
    Path _cachePathFileName;                    /** cachePath() + data name */
};
