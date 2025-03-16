#pragma once

#include "hdi/dimensionality_reduction/knn_utils.h"

#include <algorithm>
#include "Utils.h"

class TsneParameters
{
public:
    TsneParameters() :
        _knnLibrary(hdi::dr::KNN_ANNOY),
        _knnDistanceMetric(hdi::dr::KNN_METRIC_EUCLIDEAN),
        _numIterations(1000),
        _perplexity(30),
        _numTrees(4),
        _hsnw_M(16),
        _hsnw_eff(200),
        _exactKnn(false),
        _exaggerationIter(250),
        _exponentialDecayIter(150),
        _numDimensionsOutput(2),
        _exaggerationFactor(-1), // -1 means not set by user and we'll use a heuristic instead, see TsneAnalysis.cpp
        _hasPresetEmbedding(false),
        _publishExtendsAtIteration(0)   // 0 means nothing will be published
    {

    }

    void setKnnAlgorithm(hdi::dr::knn_library library) { _knnLibrary = library; _exactKnn = false; }
    void setKnnAlgorithm(utils::knn_library library) {
        if (library == utils::knn_library::KNN_EXACT)
            _exactKnn = true;

        // For possible future usage of other knn libraries
        _knnLibrary = hdi::dr::knn_library::KNN_ANNOY;
        bool isInHDILib = utils::convertToHDILibKnnLib(library, _knnLibrary);
    }

    void setKnnDistanceMetric(hdi::dr::knn_distance_metric knnDistanceMetric) { _knnDistanceMetric = knnDistanceMetric; }
    void setNumIterations(uint32_t numIterations) { _numIterations = numIterations; }
    void setPerplexity(uint32_t perplexity) { _perplexity = perplexity; }

    void setNumTrees(uint32_t numTrees) { _numTrees = numTrees; }
    void setHNSW_M(uint32_t M) { _hsnw_M = M; }
    void setHNSW_eff(uint32_t eff) { _hsnw_eff = eff; }

    void setExaggerationIter(uint32_t exaggerationIter) { _exaggerationIter = exaggerationIter; }
    void setExaggerationFactor(double exaggerationFactor) { 
        if (exaggerationFactor == -1.0)
            _exaggerationFactor = exaggerationFactor;   // -1.0 indicates automatic setting: 4 + _numPoints / 60000.0
        else
            _exaggerationFactor = std::max(exaggerationFactor, 0.0); 
    }
    void setExponentialDecayIter(uint32_t exponentialDecayIter) { _exponentialDecayIter = exponentialDecayIter; }
    void setNumDimensionsOutput(uint32_t numDimensionsOutput) { _numDimensionsOutput = numDimensionsOutput; }
    void setHasPresetEmbedding(bool hasPresetEmbedding) { _hasPresetEmbedding = hasPresetEmbedding; }
    void setPublishExtendsAtIteration(uint32_t publishExtendsAtIteration) { _publishExtendsAtIteration = publishExtendsAtIteration; }

    hdi::dr::knn_library getKnnAlgorithm() { return _knnLibrary; }
    hdi::dr::knn_distance_metric getKnnDistanceMetric() { return _knnDistanceMetric; }
    uint32_t getNumIterations() { return _numIterations; }
    uint32_t getPerplexity() { return _perplexity; }
    bool getExactKnn() const { return _exactKnn; }
    uint32_t getNumTrees() { return _numTrees; }
    uint32_t getHNSWM() { return _hsnw_M; }
    uint32_t getHNSWeff() { return _hsnw_eff; }
    uint32_t getExaggerationIter() { return _exaggerationIter; }
    double getExaggerationFactor() { return _exaggerationFactor; }
    uint32_t getExponentialDecayIter() { return _exponentialDecayIter; }
    uint32_t getNumDimensionsOutput() { return _numDimensionsOutput; }
    bool getHasPresetEmbedding() { return _hasPresetEmbedding; }
    uint32_t getPublishExtendsAtIteration() { return _publishExtendsAtIteration; }

private:
    hdi::dr::knn_library _knnLibrary;
    hdi::dr::knn_distance_metric _knnDistanceMetric;
    uint32_t _numIterations;
    uint32_t _perplexity;
    uint32_t _numTrees;
    uint32_t _hsnw_M;
    uint32_t _hsnw_eff;
    uint32_t _exaggerationIter;
    uint32_t _exponentialDecayIter;
    double _exaggerationFactor;
    uint32_t _numDimensionsOutput;
    bool _hasPresetEmbedding;
    uint32_t _publishExtendsAtIteration;

    bool _exactKnn;                 /** Compute Exact KNN instead of approximation */

};
