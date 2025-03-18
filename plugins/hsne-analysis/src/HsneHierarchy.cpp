#include "HsneHierarchy.h"

#include "HsneParameters.h"
#include "PointData/PointData.h"
#include "Utils.h"
#include "UtilsScale.h"
#include "Logger.h"

#include <nlohmann/json.hpp>

#include <iostream>
#include <fstream>
#include <algorithm>    // std::partial_sort_copy, max_element
#include <utility>      // pair
#include <numeric>
#include <mutex>
#include <execution>
#include <limits>

// set suffix strings for cache
constexpr auto _CACHE_SUBFOLDER_ = "roi-hsne-cache";
constexpr auto _HIERARCHY_CACHE_EXTENSION_ = "_hierarchy.hsne";
constexpr auto _INFLUENCE_TOPDOWN_CACHE_EXTENSION_ = "_influence-tp-hierarchy.hsne";
constexpr auto _INFLUENCE_BUTTUP_CACHE_EXTENSION_ = "_influence-bu-hierarchy.hsne";
constexpr auto _PARAMETERS_CACHE_EXTENSION_ = "_parameters.hsne";
constexpr auto _TRANSITIONNN_CACHE_EXTENSION_ = "_transitionNN.hsne";
constexpr auto _PARAMETERS_CACHE_VERSION_ = "1.0";

////////////////////
// Utility functions
////////////////////

//* Utility function to check whether a file opens correctly */
const bool fileOpens(const std::ofstream& file) {
    if (!file.is_open())
    {
        Log::error("Caching failed. File could not be opened. ");
        return false;
    }

    return true;
}

////////////////////
// InfluenceHierarchy
////////////////////

void InfluenceHierarchy::initialize(const HsneHierarchy& hierarchy)
{
    Log::info("InfluenceHierarchy::initialize: for each data point for each scale, comp the influence the respc landmarks have on it");

    // Resize the influence maps and their LandmarkMaps on each scale
    const uint32_t numDataPoints = hierarchy.getScale(0).size();
    const uint32_t numScales = hierarchy.getNumScales();

    _influenceMapTopDown.resize(numScales);
    _influenceMapBottomUp.resize(numScales);

    // scale 0, the data level: numLandmarksData == numDataPoints
    _influenceMapTopDown[0].resize(numDataPoints);
    _influenceMapBottomUp[0].resize(numDataPoints);
    const Hsne::scale_type& bottomScale = hierarchy.getScale(0);

    // for scales 1 to n
    for (uint32_t scale = 1; scale < numScales; scale++)
    {
        uint32_t numLandmarks = hierarchy.getScale(scale).size();

        _influenceMapTopDown[scale].resize(numLandmarks);
        _influenceMapBottomUp[scale].resize(numDataPoints);
    }

    // guards _influenceMapTopDown and progressCounter
    std::mutex m_influenceMapTopDown;
    std::mutex m_progressCounter;

    uint32_t progressCounter = 0;
    float progress = 0.0;

    // for each data point, get the influence of landmarks
    auto range = utils::pyrange(numDataPoints);
    std::for_each(utils::exec_policy, range.begin(), range.end(), [&](const auto i) {
        // influence[scale]: influence exercised on the data points i by landmarks in a scale 
        std::vector<std::unordered_map<uint32_t, float>> influence;

        float threshTopDown = 0.01f;
        hierarchy.getInfluenceOnDataPoint(i, influence, threshTopDown, false);

        // make sure that each data point i is influenced by at least 1 landmark in each scale
        // if not, lower the influence treshold by the factor 0.1. Tries this maximally 3 times
        {
            bool redo = false;
            const int tries_max = 3;

            for (int tries = 0; tries < tries_max; tries++)
            {
                for (uint32_t scale = 1; scale < numScales; scale++)
                {
                    if (influence[scale].size() < 1)
                    {
                        redo = true;
                        break;  // immedeatly break out the inner for loop once we know that there is a redo
                    }
                }

                // get influence for lower treshold if necessary, else break out of the loop
                if (redo)
                {
                    //if(threshTopDown < 0.0005) 
                        // Log::info("Couldn't find landmark for point " + std::to_string(i) + " at scale " + std::to_string(redo) + " num possible landmarks " + std::to_string(influence[redo].size()) + "\nSetting new threshold to " + std::to_string(threshTopDown * 0.1));
                    threshTopDown *= 0.1f;
                    hierarchy.getInfluenceOnDataPoint(i, influence, threshTopDown, false);
                }
                else
                {
                    break; // if we don't need to redo, we can break out of the trying-loop
                }
            }
        }

        // for scale 0 points only influence themselve
        _influenceMapTopDown[0][i].push_back(bottomScale._landmark_to_original_data_idx[i]);
        _influenceMapBottomUp[0][i].push_back(bottomScale._landmark_to_original_data_idx[i]);

        // for each scale above 0 
        // take the landmark that influences data point i the most and save it as
        //      _influenceMapTopDown[scale][topInfluencingLandmark].push_back(i)
        // and save the reverse mapping as
        //      _influenceMapBottomUp[scale][i].push_back(topInfluencingLandmark)
        for (uint32_t scale = 1; scale < numScales; scale++)
        {
            // order influences
            const std::unordered_map<uint32_t, float>& scaleMap = influence[scale];

            // check if there are any influcing landmarks
            if (scaleMap.size() == 0)
            {
                Log::error("Failed to find landmark for point " + std::to_string(i) + " at scale " + std::to_string(scale) + ". Num possible landmarks: " + std::to_string(scaleMap.size()));
                continue;
            }

            // find landmarks with largest influence
            auto cmpLambda = [](const std::pair<uint32_t, float>& lhs, const std::pair<uint32_t, float>& rhs) { return lhs.second < rhs.second; };
            auto largestInfluencingLandmark = std::max_element(scaleMap.begin(), scaleMap.end(), cmpLambda);

            _influenceMapBottomUp[scale][i].push_back(largestInfluencingLandmark->first);   // _influenceMapBottomUp[scale][i].size() will be at most 1

            {   // guard in case largestInfluencingLandmark->first is the same for another i on this scale
                std::scoped_lock<std::mutex> lk(m_influenceMapTopDown);
                _influenceMapTopDown[scale][largestInfluencingLandmark->first].push_back(i);
            }

        }

        // print progress
        {
            std::scoped_lock<std::mutex> lk(m_progressCounter);
            progressCounter++;
            progress = static_cast<float>(progressCounter) / numDataPoints;
            if (static_cast<uint32_t>(progress * 10000) % 10 == 0) // update every 10th percent
                std::cout << '\r' << fmt::format("Progress: {:.1f}% ({}/{})", progress*100, progressCounter, numDataPoints);   // rewrite progress line 
        }
    });
    std::cout << std::endl; // next line after progress
}


////////////////////
// HsneHierarchy  //
////////////////////

void HsneHierarchy::setParameters(const HsneParameters& parameters)
{
    _params = parameters.getHDILibHsneParams();
}


void HsneHierarchy::initialize(mv::CoreInterface* core, const Points& inputData, const std::vector<bool>& enabledDimensions, const HsneParameters& parameters, const std::string& cachePath)
{
    _core = core;

    // Convert our own HSNE parameters to the HDI parameters
    setParameters(parameters);

    // Extract the enabled dimensions from the data
    _numDimensions = std::count_if(enabledDimensions.begin(), enabledDimensions.end(), [](bool b) { return b; });

    _numScales = parameters.getNumScales();
    _numPoints = inputData.getNumPoints();
    _inputDataName = inputData.getGuiName();
    _exactKnn = parameters.getExactKnn();

    assert(_numScales > 0);
    if (_numScales <= 0)
        Log::warn("HsneHierarchy::initialize: _numScales must be > 0.");

    // set cachePath if given
    if (cachePath == std::string())
        _cachePath = std::filesystem::current_path() / _CACHE_SUBFOLDER_;
    else
        _cachePath = std::filesystem::path(cachePath) / _CACHE_SUBFOLDER_;

    _cachePathFileName = _cachePath / _inputDataName.toStdString();

    // Initialize hierarchy
    _hsne = std::make_unique<Hsne>();
    _log = std::make_unique<hdi::utils::CoutLog>();

    // Check of hsne data can be loaded from cache on disk, otherwise compute hsne hierarchy
    bool hsneLoadedFromCache = loadCache();
    if (hsneLoadedFromCache == false) {
        Log::info("HsneHierarchy::initialize() compute HSNE hierarchy.");

        // Set up a logger
        _hsne->setLogger(_log.get());

        // Set the dimensionality of the data in the HSNE object
        _hsne->setDimensionality(_numDimensions);

        Log::redirect_std_io_to_logger();

        // Init hierarchy, discard local data copy afterwards
        {
            // Get data from core
            std::vector<float> data;
            std::vector<uint32_t> dimensionIndices;

            data.resize((inputData.isFull() ? inputData.getNumPoints() : inputData.indices.size()) * _numDimensions);
            for (uint32_t i = 0; i < inputData.getNumDimensions(); i++)
                if (enabledDimensions[i]) dimensionIndices.push_back(i);

            inputData.populateDataForDimensions<std::vector<float>, std::vector<uint32_t>>(data, dimensionIndices);

            // Initialize HSNE with the input data and the given parameters
            if (_exactKnn)
            {
                computeSimilarities(data);
                _hsne->initialize(_similarities, _params);
            }
            else
                _hsne->initialize((Hsne::scalar_type*)data.data(), _numPoints, _params);
        }

        // Add a number of scales as indicated by the user
        for (uint32_t s = 0; s < _numScales - 1; ++s) {
            _hsne->addScale();
        }

        Log::reset_std_io();

        _influenceHierarchy.initialize(*this);

        computeTransitionNN();

        // Write HSNE hierarchy to disk
        saveCacheHsne();
    }

}

void HsneHierarchy::getTransitionMatrixForSelectionAtScale(const uint32_t scale, const uint32_t threshConnections, std::vector<uint32_t>& landmarkIdxs, HsneMatrix& transitionMatrix, float thresh) const
{
    // Get full transition matrix of the previous scale
    HsneMatrix& fullTransitionMatrix = _hsne->scale(scale)._transition_matrix;

    // Extract the selected subportion of the transition matrix
    utils::extractSubGraph(fullTransitionMatrix, threshConnections, landmarkIdxs, transitionMatrix, thresh);
}

void HsneHierarchy::getTransitionMatrixForSelectionAtScale(const uint32_t scale, std::vector<uint32_t>& landmarkIdxs, HsneMatrix& transitionMatrix) const
{
    // Get full transition matrix of the previous scale
    HsneMatrix& fullTransitionMatrix = _hsne->scale(scale)._transition_matrix;

    // Extract the selected subportion of the transition matrix
    std::vector<unsigned int> dummy;
    hdi::utils::extractSubGraph(fullTransitionMatrix, landmarkIdxs, transitionMatrix, dummy, 1);
}

void HsneHierarchy::getLocalIDsInCoarserScale(uint32_t currentScale, const std::vector<uint32_t>& landmarkIdxs, std::vector<uint32_t>& coarserScaleIDxs, float tresh /*default  = 0.5 */) const
{
    if (currentScale >= _numScales) return;

    std::map<uint32_t, float> neighborsCoarser;
    getInfluencingLandmarksInCoarserScale(currentScale, landmarkIdxs, neighborsCoarser);

    coarserScaleIDxs.clear();
    for (const auto& n : neighborsCoarser) {
        if (n.second > tresh)
        {
            coarserScaleIDxs.push_back(n.first);
        }
    }

}

void HsneHierarchy::getLocalIDsInRefinedScale(uint32_t currentScale, const std::vector<uint32_t>& landmarkIdxs, std::vector<uint32_t>& refinedScaleIDxs, float tresh /*default  = 0.5 */) const
{
    if (currentScale == 0) return;

    std::map<uint32_t, float> neighborsRefined;
    getInfluencedLandmarksInRefinedScale(currentScale, landmarkIdxs, neighborsRefined);

    refinedScaleIDxs.clear();
    for (const auto& n : neighborsRefined) {
        if (n.second > tresh)
        {
            refinedScaleIDxs.push_back(n.first);
        }
    }
}


void HsneHierarchy::computeSelectionMapsAtScale(const uint32_t scale, const std::vector<uint32_t>& localIDsOnNewScale, LandmarkMapSingle& mappingBottomToLocal, LandmarkMap& mappingLocalToBottom) const {

    // INFO: No need to ensure uniqueness in mappingLocalToBottom since InteractiveHsnePlugin::selectionMapping will take care of that

    // clear old mappings
    mappingLocalToBottom.clear();
    mappingBottomToLocal.clear();

    // Prevent rehashing when inserting values
    mappingLocalToBottom.resize(localIDsOnNewScale.size());
    mappingBottomToLocal.resize(_numPoints, std::numeric_limits<uint32_t>::max());  // use max as an indicator for no mapped value

    // A LandmarkMap is nothing but std::vector<std::vector<uint32_t>>
    // The LandmarkMap vector is of the size of numLandmarks on a given scale
    // landmarkMap[i] is a vector of data points (global IDs) on which the landmark i on a given scale (here topScaleIndex) has the highest influence
    const LandmarkMap& landmarkMapTopDown = getInfluenceHierarchy().getMapTopDown()[scale];

    auto range = utils::pyrange(static_cast<uint32_t>(localIDsOnNewScale.size()));
    std::for_each(utils::exec_policy, range.begin(), range.end(), [&](const auto posInEmbedding) {
        //const uint32_t localIdOnScale = localIDsOnNewScale[posInEmbedding];
        //const std::vector<uint32_t>& newBottomIDs = landmarkMapTopDown[localIdOnScale];

        // when selecting in the embedding, select all data level IDs that are influenced by the landmark selection
        mappingLocalToBottom[posInEmbedding] = landmarkMapTopDown[localIDsOnNewScale[posInEmbedding]];

        // for heuristic, each image point maps to one landmark
        for (auto const& bottomID : mappingLocalToBottom[posInEmbedding])
            mappingBottomToLocal[bottomID] = posInEmbedding;

        });
}

void HsneHierarchy::saveCacheHsne() const {
    if (!_hsne) return; // only save if initialize() has been called

    if (!std::filesystem::exists(_cachePath))
        std::filesystem::create_directory(_cachePath);

    Log::info("HsneHierarchy::saveCacheHsne(): save cache to " + _cachePathFileName.string());

    saveCacheHsneHierarchy(_cachePathFileName.string() + _HIERARCHY_CACHE_EXTENSION_);
    saveCacheHsneInfluenceHierarchy(_cachePathFileName.string() + _INFLUENCE_TOPDOWN_CACHE_EXTENSION_, _influenceHierarchy.getMapTopDown());
    saveCacheHsneInfluenceHierarchy(_cachePathFileName.string() + _INFLUENCE_BUTTUP_CACHE_EXTENSION_, _influenceHierarchy.getMapBottomUp());
    saveCacheParameters(_cachePathFileName.string() + _PARAMETERS_CACHE_EXTENSION_);
    saveCacheHsneTransitionNNOnScale(_cachePathFileName.string() + _TRANSITIONNN_CACHE_EXTENSION_);
}

void HsneHierarchy::saveCacheHsneHierarchy(std::string fileName) const {
    Log::info("Writing " + fileName);

    std::ofstream saveFile(fileName, std::ios::out | std::ios::binary);

    if (!fileOpens(saveFile)) return;

    hdi::dr::IO::saveHSNE(*_hsne, saveFile, _hsne->logger());

    saveFile.close();

}

void HsneHierarchy::saveCacheHsneInfluenceHierarchy(std::string fileName, const std::vector<LandmarkMap>& influenceHierarchy) const {
    Log::info("Writing " + fileName);

    std::ofstream saveFile(fileName, std::ios::out | std::ios::binary);

    if (!fileOpens(saveFile)) return;

    size_t iSize = influenceHierarchy.size();

    saveFile.write((const char*)&iSize, sizeof(decltype(iSize)));
    for (size_t i = 0; i < iSize; i++)
    {
        size_t jSize = influenceHierarchy[i].size();
        saveFile.write((const char*)&jSize, sizeof(decltype(jSize)));
        for (size_t j = 0; j < jSize; j++)
        {
            size_t kSize = influenceHierarchy[i][j].size();
            saveFile.write((const char*)&kSize, sizeof(decltype(kSize)));
            if (kSize > 0)
            {
                saveFile.write((const char*)&influenceHierarchy[i][j].front(), kSize * sizeof(uint32_t));
            }
        }
    }

    saveFile.close();
}

void HsneHierarchy::saveCacheHsneTransitionNNOnScale(std::string fileName) const {
    Log::info("Writing " + fileName);

    std::ofstream saveFile(fileName, std::ios::out | std::ios::binary);

    if (!fileOpens(saveFile)) return;

    // type of _transitionNNOnScale is 
    // std::vector<std::vector<std::vector<uint32_t>>>
    //     scales     landmarks      kNN      ID (local on scale)

    size_t iSize = _transitionNNOnScale.size();
    saveFile.write((const char*)&iSize, sizeof(decltype(iSize)));  // write number of scales
    for (size_t i = 0; i < iSize; i++) // scales
    {
        size_t jSize = _transitionNNOnScale[i].size();
        saveFile.write((const char*)&jSize, sizeof(decltype(jSize)));  // write number of landmarks on scale

        for (size_t j = 0; j < jSize; j++) // landmarks on scale
        {
            size_t kSize = _transitionNNOnScale[i][j].size();
            saveFile.write((const char*)&kSize, sizeof(decltype(kSize)));    // write number of kNN of landmarks on scale

            if (kSize > 0)
            {
                saveFile.write((const char*)&_transitionNNOnScale[i][j].front(), kSize * sizeof(uint32_t));
            }// kNN of landmarks on scale
        } // landmarks on scale
    } // scales

    saveFile.close();
}

void HsneHierarchy::saveCacheParameters(std::string fileName) const {
    Log::info("Writing " + fileName);

    std::ofstream saveFile(fileName, std::ios::out | std::ios::trunc);
    
    if (!fileOpens(saveFile)) return;

    // store parameters in json file
    nlohmann::json parameters;
    parameters["## VERSION ##"] = _PARAMETERS_CACHE_VERSION_;

    parameters["Input data name"] = _inputDataName.toStdString();
    parameters["Number of points"] = _numPoints;
    parameters["Number of dimensions"] = _numDimensions;

    parameters["Number of Scales"] = _numScales;

    parameters["Knn library"] = _params._aknn_algorithm;
    parameters["Knn exact"] = _exactKnn;
    parameters["Knn distance metric"] = _params._aknn_metric;
    parameters["Knn number of neighbors"] = _params._num_neighbors;

    parameters["Nr. Trees for AKNN (Annoy)"] = _params._aknn_annoy_num_trees;
    parameters["Parameter M (HNSW)"] = _params._aknn_hnsw_M;
    parameters["Parameter eff (HNSW)"] = _params._aknn_hnsw_eff;

    parameters["Memory preserving computation"] = _params._out_of_core_computation;
    parameters["Nr. RW for influence"] = _params._num_walks_per_landmark;
    parameters["Nr. RW for Monte Carlo"] = _params._mcmcs_num_walks;
    parameters["Random walks threshold"] = _params._mcmcs_landmark_thresh;
    parameters["Random walks length"] = _params._mcmcs_walk_length;
    parameters["Pruning threshold"] = _params._transition_matrix_prune_thresh;
    parameters["Fixed Percentile Landmark Selection"] = _params._hard_cut_off;
    parameters["Percentile Landmark Selection"] = _params._hard_cut_off_percentage;

    parameters["Seed for random algorithms"] = _params._seed;
    parameters["Select landmarks with a MCMCS"] = _params._monte_carlo_sampling;

    // Write to file
    saveFile << std::setw(4) << parameters << std::endl;
    saveFile.close();
}

bool HsneHierarchy::loadCache() {
    Log::info("HsneHierarchy::loadCache(): attempt to load cache from " + _cachePathFileName.string());

    auto pathParameter = _cachePathFileName.string() + _PARAMETERS_CACHE_EXTENSION_;
    auto pathHierarchy = _cachePathFileName.string() + _HIERARCHY_CACHE_EXTENSION_;
    auto pathInfluenceTD = _cachePathFileName.string() + _INFLUENCE_TOPDOWN_CACHE_EXTENSION_;
    auto pathInfluenceBU = _cachePathFileName.string() + _INFLUENCE_BUTTUP_CACHE_EXTENSION_;
    auto pathTransition = _cachePathFileName.string() + _TRANSITIONNN_CACHE_EXTENSION_;
   
    for (const Path& path : { pathHierarchy, pathInfluenceTD, pathInfluenceBU, pathParameter, pathTransition })
    {
        if (!(std::filesystem::exists(path)))
        {
            Log::info("Loading cache failed: No file exists at: " + path.string());
            return false;
        }
    }
    
    if (!checkCacheParameters(pathParameter))
    {
        Log::warn("Loading cache failed: Current settings are different from cached parameters.");
        return false;
    }

    auto checkCache = [](bool success, std::string path) -> bool
    {
        if (success)
            return true;
        else
        {
            Log::error("Loading cache failed: " + path);
            return false;
        }
    };

    if (!checkCache(loadCacheHsneHierarchy(pathHierarchy), pathHierarchy))
        return false;

    if (!checkCache(loadCacheHsneInfluenceHierarchy(pathInfluenceTD, _influenceHierarchy.getMapTopDown()), pathInfluenceTD))
        return false;

    if (!checkCache(loadCacheHsneInfluenceHierarchy(pathInfluenceBU, _influenceHierarchy.getMapBottomUp()), pathInfluenceBU))
        return false;

    if (!checkCache(loadCacheHsneTransitionNNOnScale(pathTransition), pathTransition))
        return false;

    Log::info("HsneHierarchy::loadCache: loading hierarchy from cache was successfull");

    return true;
}

bool HsneHierarchy::loadCacheHsneHierarchy(std::string fileName) {
    std::ifstream loadFile(fileName.c_str(), std::ios::in | std::ios::binary);

    if (!loadFile.is_open()) return false;

    Log::info("Loading " + fileName);

    if (_hsne) { 
        _hsne.reset();
        _hsne = std::make_unique<Hsne>();
    }

    Log::redirect_std_io_to_logger();

    _hsne->setLogger(_log.get());

    // TODO: check if hsne matches data
    hdi::dr::IO::loadHSNE(*_hsne, loadFile, _log.get());

    _numScales = static_cast<uint32_t>(_hsne->hierarchy().size());

    Log::reset_std_io();

    return true;

}

bool HsneHierarchy::loadCacheHsneInfluenceHierarchy(std::string fileName, std::vector<LandmarkMap>& influenceHierarchy) {
    if (!_hsne) return false;

    std::ifstream loadFile(fileName.c_str(), std::ios::in | std::ios::binary);

    if (!loadFile.is_open()) return false;

    Log::info("Loading " + fileName);

    // TODO: check if hsne matches data
    size_t iSize = 0;
    loadFile.read((char*)&iSize, sizeof(decltype(iSize)));

    influenceHierarchy.resize(iSize);

    for (size_t i = 0; i < influenceHierarchy.size(); i++)
    {
        size_t jSize = 0;
        loadFile.read((char*)&jSize, sizeof(decltype(jSize)));
        influenceHierarchy[i].resize(jSize);

        for (size_t j = 0; j < influenceHierarchy[i].size(); j++)
        {
            size_t kSize = 0;
            loadFile.read((char*)&kSize, sizeof(decltype(kSize)));

            influenceHierarchy[i][j].resize(kSize);
            if (kSize > 0)
            {
                loadFile.read((char*)&influenceHierarchy[i][j].front(), kSize * sizeof(uint32_t));
            }
        }
    }

    loadFile.close();

    return true;

}

bool HsneHierarchy::loadCacheHsneTransitionNNOnScale(std::string fileName) {
    if (!_hsne) return false;

    std::ifstream loadFile(fileName.c_str(), std::ios::in | std::ios::binary);

    if (!loadFile.is_open()) return false;

    Log::info("Loading " + fileName);

    // type of _transitionNNOnScale is 
    // std::vector<std::vector<std::vector<uint32_t>>>
    //     scales     landmarks      kNN    ID (local on scale)

    // TODO: check if hsne matches data
    size_t iSize = 0;
    loadFile.read((char*)&iSize, sizeof(decltype(iSize)));  // load number of scales
    _transitionNNOnScale.resize(iSize);

    for (size_t i = 0; i < _transitionNNOnScale.size(); i++)
    {
        size_t jSize = 0;
        loadFile.read((char*)&jSize, sizeof(decltype(jSize)));  // load number of landmarks on scale
        _transitionNNOnScale[i].resize(jSize);

        for (size_t j = 0; j < _transitionNNOnScale[i].size(); j++)
        {
            size_t kSize = 0;
            loadFile.read((char*)&kSize, sizeof(decltype(kSize)));  // load number of kNN of landmarks on scale

            _transitionNNOnScale[i][j].resize(kSize);
            if (kSize > 0)
            {
                loadFile.read((char*)&_transitionNNOnScale[i][j].front(), kSize * sizeof(uint32_t));
            }
        }
    }

    loadFile.close();

    return true;

}

bool HsneHierarchy::checkCacheParameters(std::string fileName) const {
    if (!_hsne) return false;

    std::ifstream loadFile(fileName.c_str(), std::ios::in);

    if (!loadFile.is_open()) return false;

    Log::info("Loading " + fileName);

    // read a JSON file
    nlohmann::json parameters;
    loadFile >> parameters;

    if (!parameters.contains("## VERSION ##") || parameters["## VERSION ##"] != _PARAMETERS_CACHE_VERSION_)
    {
        Log::info("Version of the cache (" + std::string(parameters["## VERSION ##"]) + ") differs from analysis version (" + _PARAMETERS_CACHE_VERSION_ + "). Cannot load cache)");
        return false;
    }

    // if current setting is different from params on disk, don't load from disk
    auto checkParam = [&parameters](std::string paramName, auto localParam) -> bool {
        const auto& storedParam = parameters[paramName];
        if (storedParam != localParam)
        {
            std::ostringstream localParamSS, storedParamSS;
            localParamSS << localParam;
            storedParamSS << storedParam;
            Log::info(paramName + " (" + localParamSS.str() + ") does not match cache (" + storedParamSS.str() + "). Cannot load cache.");
            return false;
        }
        return true;
    };

    if (!checkParam("Input data name", _inputDataName.toStdString()) ) return false;
    if (!checkParam("Number of points", _numPoints) ) return false;
    if (!checkParam("Number of dimensions", _numDimensions) ) return false;

    if (!checkParam("Number of Scales", _numScales) ) return false;

    if (!checkParam("Knn library", _params._aknn_algorithm) ) return false;
    if (!checkParam("Knn distance metric", _params._aknn_metric) ) return false;

    // only check "Knn exact" if it's in the parameters file
    if (!parameters["Knn exact"].is_null())
        checkParam("Knn exact", _exactKnn);

    if (!checkParam("Knn number of neighbors", _params._num_neighbors) ) return false;

    if (!checkParam("Nr. Trees for AKNN (Annoy)", _params._aknn_annoy_num_trees) ) return false;
    if (!checkParam("Parameter M (HNSW)", _params._aknn_hnsw_M) ) return false;
    if (!checkParam("Parameter eff (HNSW)", _params._aknn_hnsw_eff) ) return false;

    if (!checkParam("Memory preserving computation", _params._out_of_core_computation) ) return false;
    if (!checkParam("Nr. RW for influence", _params._num_walks_per_landmark) ) return false;
    if (!checkParam("Nr. RW for Monte Carlo", _params._mcmcs_num_walks) ) return false;
    if (!checkParam("Random walks threshold", _params._mcmcs_landmark_thresh) ) return false;
    if (!checkParam("Random walks length", _params._mcmcs_walk_length) ) return false;
    if (!checkParam("Pruning threshold", _params._transition_matrix_prune_thresh) ) return false;
    if (!checkParam("Fixed Percentile Landmark Selection", _params._hard_cut_off) ) return false;
    if (!checkParam("Percentile Landmark Selection", _params._hard_cut_off_percentage) ) return false;

    if (!checkParam("Seed for random algorithms", _params._seed) ) return false;
    if (!checkParam("Select landmarks with a MCMCS", _params._monte_carlo_sampling) ) return false;

    Log::info("Parameters of cache correspond to current settings.");

    return true;
}

void HsneHierarchy::computeTransitionNN() {
    Log::info("HsneHierarchy: computeTransitionNN");

    // type of _transitionNNOnScale is 
    // std::vector<std::vector<std::vector<uint32_t>>>
    //     scales     landmarks      kNN     ID (local on scale)
    _transitionNNOnScale.resize(_numScales);

    const size_t nn = _params._num_neighbors;                                   // use same number of neighbors as t-SNE

    auto range = utils::pyrange(_numScales);
    std::for_each(utils::exec_policy, range.begin(), range.end(), [&](const auto i) {
        HsneMatrix& fullTransitionMatrix = getScale(i)._transition_matrix;      // transition matrix for landmarks at scale i
        std::vector<std::vector<uint32_t>>& nnOnScale = _transitionNNOnScale[i];

        nnOnScale.reserve(fullTransitionMatrix.size());                         // resize with the number of landmarks on the scale

        transitionVec sortedNN(nn);

        for (auto& transitionValues : fullTransitionMatrix)                     // the MapMemEff is basically a std::vector<std::pair<Key,T>>
        {
            transitionVec temp_vec;
            for (Eigen::SparseVector<float>::InnerIterator it(transitionValues.memory()); it; ++it) {
                temp_vec.emplace_back(it.index(), it.value());
            }

            // sort values and store in sortedNN
            std::partial_sort_copy(temp_vec.begin(), temp_vec.end(),
                sortedNN.begin(), sortedNN.end(),
                [](const std::pair<uint32_t, float>& a, const std::pair<uint32_t, float>& b) {return a.second > b.second; });

            // copy first IDs entries of sortedNN and store in sortedIDs
            std::vector<uint32_t> sortedIDs;
            for (const auto& [sortedID, sortedVal] : sortedNN)
                sortedIDs.push_back(sortedID);

            // save IDs
            nnOnScale.push_back(sortedIDs);
        }
        });

}

void HsneHierarchy::computeSimilarities(const std::vector<float>& data)
{
    Log::info("HsneHierarchy::computeSimilarities for exact nearest neighbors");
    utils::ScopedTimer computeSimilaritiesTimer("Total time computing similarities (including knn)");
    std::vector<float> distance_based_probabilities;
    std::vector<uint32_t> neighborhood_graph;

    size_t nn = static_cast<size_t>(_params._num_neighbors) + 1;

    utils::timer([&]() {
        utils::computeExactKNN(data, data, _numPoints, _numPoints, _numDimensions, nn, distance_based_probabilities, neighborhood_graph);
        },
    "computeExactKNN");

    utils::computeFMC(_numPoints, nn, distance_based_probabilities, neighborhood_graph);
    utils::computeSimilaritiesFromKNN(distance_based_probabilities, neighborhood_graph, _numPoints, _similarities);

}
