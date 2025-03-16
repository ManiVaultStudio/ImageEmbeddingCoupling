#pragma once

#include "actions/GroupAction.h"
#include "actions/IntegralAction.h"
#include "actions/DecimalAction.h"
#include "actions/ToggleAction.h"
#include "actions/OptionAction.h"

using namespace mv::gui;

class QMenu;
class HsneSettingsAction;

/**
 * Advanced HSNE setting action class
 *
 * Action class for advanced HSNE settings
 *
 * @author Thomas Kroes
 */
class AdvancedHsneSettingsAction : public GroupAction
{
public:

    /**
     * Constructor
     * @param hsneSettingsAction Reference to HSNE settings action
     */
    AdvancedHsneSettingsAction(HsneSettingsAction& hsneSettingsAction);

public: // Action getters

    HsneSettingsAction& getHsneSettingsAction() { return _hsneSettingsAction; }
    IntegralAction& getNumWalksForLandmarkSelectionAction() { return _numWalksForLandmarkSelectionAction; }
    DecimalAction& getNumWalksForLandmarkSelectionThresholdAction() { return _numWalksForLandmarkSelectionThresholdAction; }
    IntegralAction& getRandomWalkLengthAction() { return _randomWalkLengthAction; }
    IntegralAction& getNumWalksForAreaOfInfluenceAction() { return _numWalksForAreaOfInfluenceAction; }
    IntegralAction& getMinWalksRequiredAction() { return _minWalksRequiredAction; }
    IntegralAction& getNumTreesAknnAction() { return _numTreesAknnAction; }
    IntegralAction& getHNSWMAction() { return _HNSW_M_Action; }
    IntegralAction& getHNSWeffAction() { return _HNSW_eff_Action; }
    ToggleAction& getHardCutOffAction() { return _hardCutOffAction; }
    DecimalAction& getHardCutOffPercentageAction() { return _hardCutOffPercentageAction; }
    ToggleAction& getUseOutOfCoreComputationAction() { return _useOutOfCoreComputationAction; }
    ToggleAction& getInitWithPCA() { return _initWithPCAAction; }

    /* "SVD" = 0, "COV" = 1 (default) */
    OptionAction& getPcaAlgorithmAction() { return _pcaAlgorithmAction; }

protected:
    HsneSettingsAction&     _hsneSettingsAction;                                /** Reference to HSNE settings action */
    IntegralAction          _numWalksForLandmarkSelectionAction;                /** Number of walks for landmark selection action */
    DecimalAction           _numWalksForLandmarkSelectionThresholdAction;       /** Number of walks for landmark selection threshold action */
    IntegralAction          _randomWalkLengthAction;                            /** Random walk length action */
    IntegralAction          _numWalksForAreaOfInfluenceAction;                  /** Number of walks for area of influence action */
    IntegralAction          _minWalksRequiredAction;                            /** Minimum number of walks required action */
    IntegralAction          _numTreesAknnAction;                                /** Annoy: Number of KNN trees action */
    IntegralAction          _HNSW_M_Action;                                     /** HNSW: M */
    IntegralAction          _HNSW_eff_Action;                                   /** HNSW: eff */
    ToggleAction            _useOutOfCoreComputationAction;                     /** Use out of core computation action */
    ToggleAction            _initWithPCAAction;                                 /** Init first embedding with PCA */
    ToggleAction            _hardCutOffAction;                                  /** Select landmarks based on a user provided hard percentage cut off, instead of data-driven */
    DecimalAction           _hardCutOffPercentageAction;                        /** percentage of previous level landmarks to use in next level when using the hard cut off */
    OptionAction            _pcaAlgorithmAction;                                /** PCA algorithm action */
};
