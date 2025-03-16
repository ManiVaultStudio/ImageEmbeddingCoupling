#pragma once

#include "hdi/dimensionality_reduction/knn_utils.h"
#include "CommonTypes.h"
#include "Utils.h"

/**
 * HsneParameters
 *
 * Container class for the parameters associated with the HSNE algorithm
 *
 * @author Alexander Vieth
 */
class HsneParameters
{
public:
    HsneParameters() :
        _hdi_hsne_params(),
        _numScales(3),
        _initWithPCA(false),
        _exactKnn(false)
    {

    }

    /// SETTER

    void setKnnLibrary(hdi::dr::knn_library library) { _hdi_hsne_params._aknn_algorithm = library; _exactKnn = false; }
    void setKnnLibrary(utils::knn_library library) { 
        if (library == utils::knn_library::KNN_EXACT)
            _exactKnn = true;

        // For possible future usage of other knn libraries
        _hdi_hsne_params._aknn_algorithm = hdi::dr::knn_library::KNN_ANNOY;
        bool isInHDILib = utils::convertToHDILibKnnLib(library, _hdi_hsne_params._aknn_algorithm);        
    }
    void setNumScales(uint32_t numScales) { 
        if (numScales <= 0)
        {
            _numScales = 1;
            Log::error("HsneParameters::setNumScales: numScales must be > 0");
        }
        _numScales = numScales; 
    }
    void setSeed(int seed) { _hdi_hsne_params._seed = seed; }

    /* In HDILib:hierarchical_sne_inl.h some values are set based on _num_neighbors
       perplexity = _num_neighbors / 3
       nn = _num_neighbors + 1
    */
    void setNNWithPerplexity(uint32_t perplexity) { _hdi_hsne_params._num_neighbors = perplexity * 3; }

    void setNumWalksForLandmarkSelection(uint32_t numWalks) { _hdi_hsne_params._mcmcs_num_walks = numWalks; }
    void setNumWalksForLandmarkSelectionThreshold(float landmarkThresh) { _hdi_hsne_params._mcmcs_landmark_thresh = landmarkThresh; }
    void setRandomWalkLength(uint32_t length) { _hdi_hsne_params._mcmcs_walk_length = length; }
    void setNumWalksForAreaOfInfluence(uint32_t numWalks) { _hdi_hsne_params._num_walks_per_landmark = numWalks; }
    void setMinWalksRequired(uint32_t minWalks) { _hdi_hsne_params._transition_matrix_prune_thresh = static_cast<float>(minWalks); }
    void useMonteCarloSampling(bool useMonteCarloSampling) { _hdi_hsne_params._monte_carlo_sampling = useMonteCarloSampling; }
    void useOutOfCoreComputation(bool useOutOfCoreComputation) { _hdi_hsne_params._out_of_core_computation = useOutOfCoreComputation; }
    void initWithPCA(bool initWithPCA) { _initWithPCA = initWithPCA; }
    void setAknnMetric(hdi::dr::knn_distance_metric aknn_metric) { _hdi_hsne_params._aknn_metric = aknn_metric; }
    void setHardCutOff(bool hard_cut_off) { _hdi_hsne_params._hard_cut_off = hard_cut_off; }
    void setHardCutOffPercentage(float hard_cut_off_percentage) { _hdi_hsne_params._hard_cut_off_percentage = hard_cut_off_percentage; }
    void setRSReductionFactorPerLayer(float rs_reduction_factor_per_layer) { _hdi_hsne_params._rs_reduction_factor_per_layer = rs_reduction_factor_per_layer; }
    void setRSOutlierRemovalJumps(uint32_t rs_outliers_removal_jumps) { _hdi_hsne_params._rs_outliers_removal_jumps = rs_outliers_removal_jumps; }
    void setNumTreesAKNN(uint32_t numTrees) { _hdi_hsne_params._aknn_annoy_num_trees = numTrees; }
    void setHNSW_M(uint32_t M) { _hdi_hsne_params._aknn_hnsw_M = M; }
    void setHNSW_eff(uint32_t eff) { _hdi_hsne_params._aknn_hnsw_eff = eff; }


    // GETTER

    Hsne::Parameters getHDILibHsneParams() const { return _hdi_hsne_params; }

    /** Enum specifying which approximate nearest neighbour library to use for the similarity computation */
    hdi::dr::knn_library getKnnLibrary() const { return _hdi_hsne_params._aknn_algorithm; }
    /** Number of scales the hierarchy should consist of */
    uint32_t getNumScales() const { return _numScales; }
    /** Seed used for random algorithms. If a negative value is provided a time based seed is used */
    int getSeed() const { return _hdi_hsne_params._seed; }
    /** Number of neighbors used in the KNN graph */
    uint32_t getNN() const { return _hdi_hsne_params._num_neighbors; }
    /** Perplexity value */
    float getPerplexity() const { return _hdi_hsne_params._num_neighbors / 3.f; }

    /** Whether exact or approximated knn are used */
    bool getExactKnn() const { return _exactKnn; }

    /** How many random walks to use in the MCMC sampling process */
    uint32_t getNumWalksForLandmarkSelection() const { return _hdi_hsne_params._mcmcs_num_walks; }
    /** How many times a landmark should be the endpoint of a random walk before it is considered a landmark */
    float getNumWalksForLandmarkSelectionThreshold() const { return _hdi_hsne_params._mcmcs_landmark_thresh; }
    /** How long each random walk should be */
    uint32_t getRandomWalkLength() const { return _hdi_hsne_params._mcmcs_walk_length; }
    /** How many random walks to use for computing the area of influence */
    uint32_t getNumWalksForAreaOfInfluence() const { return _hdi_hsne_params._num_walks_per_landmark; }
    /** Minimum number of walks to be considered in the computation of the transition matrix */
    uint32_t getMinWalksRequired() const { return static_cast<uint32_t>(_hdi_hsne_params._transition_matrix_prune_thresh); }
    /** Whether to use markov chain monte carlo sampling for computing the landmarks */
    bool useMonteCarloSampling() const { return _hdi_hsne_params._monte_carlo_sampling; }
    /** Preserve memory while computing the hierarchy */
    bool useOutOfCoreComputation() const { return _hdi_hsne_params._out_of_core_computation; }
    /** Initialize embeddings with PCA */
    bool initWithPCA() const { return _initWithPCA; }
    /** Knn distance metric */
    hdi::dr::knn_distance_metric getAknnMetric() const { return _hdi_hsne_params._aknn_metric; }

    /** Nr. Trees for AKNN (Annoy) */
    uint32_t getNumTreesAKNN() const { return _hdi_hsne_params._aknn_annoy_num_trees; }
    /** M Parameter (HNSW) */
    uint32_t getHNSW_M() const { return static_cast<uint32_t>(_hdi_hsne_params._aknn_hnsw_M); }
    /** Eff Parameter (HNSW) */
    uint32_t getHNSW_eff() const { return static_cast<uint32_t>(_hdi_hsne_params._aknn_hnsw_eff); }

    /** Select landmarks based on a user provided hard percentage cut off, instead of data-driven */
    bool getHardCutOff() const { return _hdi_hsne_params._hard_cut_off; }
    /** percentage of previous level landmarks to use in next level when using the hard cut off*/
    float getHardCutOffPercentage() const { return _hdi_hsne_params._hard_cut_off_percentage; }
    /** Reduction factor per layer used in the random sampling (RS) */
    float getRSReductionFactorPerLayer() const { return _hdi_hsne_params._rs_reduction_factor_per_layer; }
    /** Random walks used in the RS to avoid outliers */
    uint32_t getRSOutlierRemovalJumps() const { return _hdi_hsne_params._rs_outliers_removal_jumps; }


private:
    /** HDILib HSNE parameters */
    Hsne::Parameters    _hdi_hsne_params;

    // additional params outside HDI Lib

    uint32_t _numScales;        /** Number of scales the hierarchy should consist of */
    bool _initWithPCA;              /** Initialize embeddings with PCA */
    bool _exactKnn;                 /** Compute Exact KNN instead of approximation */

};
