#pragma once

#include "actions/GroupAction.h"
#include "actions/IntegralAction.h"
#include "actions/ToggleAction.h"
#include "actions/OptionAction.h"
#include "actions/TriggerAction.h"

#include "Utils.h"

#include <QMap> 

using namespace mv::gui;

class HsneSettingsAction;

/// ////////////// ///
/// TsneROIActions ///
/// ////////////// ///

class TsneROIActions : public WidgetAction
{
protected:

    /** Widget class for TSNE computation action */
    class Widget : public WidgetActionWidget {
    public:

        /**
         * Constructor
         * @param parent Pointer to parent widget
         * @param tsneComputationAction Pointer to TSNE computation action
         */
        Widget(QWidget* parent, TsneROIActions* tsneROIActions);
    };

    /**
     * Get widget representation of the TSNE computation action
     * @param parent Pointer to parent widget
     * @param widgetFlags Widget flags for the configuration of the widget (type)
     */
    QWidget* getWidget(QWidget* parent, const std::int32_t& widgetFlags) override {
        return new Widget(parent, this);
    };

public:

    /**
     * Constructor
     * @param parent Pointer to parent object
     */
    TsneROIActions(QObject* parent);

public: // Action getters

    TriggerAction& getTSNERoiAction() { return _tsneROIAction; }
    TriggerAction& getTSNELandmarkAction() { return _tsneLandmarkROIAction; }

protected:
    TriggerAction           _tsneROIAction;                     /** Start ROI tSNE action */
    TriggerAction           _tsneLandmarkROIAction;             /** Start ROI landmarks tSNE action */
};


/// //////////////////// ///
/// General HSNE setting ///
/// //////////////////// ///

/**
 * General HSNE setting action class
 *
 * Actions class for general HSNE settings
 *
 * @author Thomas Kroes
 */
class GeneralHsneSettingsAction : public GroupAction
{
public:

    /**
     * Constructor
     * @param hsneSettingsAction Reference to HSNE settings action
     */
    GeneralHsneSettingsAction(HsneSettingsAction& hsneSettingsAction);

public: // Action getters

    HsneSettingsAction& getHsneSettingsAction() { return _hsneSettingsAction; }
    OptionAction& getKnnTypeAction() { return _knnTypeAction; }
    OptionAction& getDistanceMetricAction() { return _distanceMetricAction; };
    IntegralAction& getPerplexityAction() { return _perplexityAction; };
    IntegralAction& getNumScalesAction() { return _numScalesAction; }
    IntegralAction& getSeedAction() { return _seedAction; }
    ToggleAction& getUseMonteCarloSamplingAction() { return _useMonteCarloSamplingAction; }
    TriggerAction& getInitAction() { return _initAction; }
    TsneROIActions& getTSNERoiActionsGroup() { return _tsneROIActions; }
    TriggerAction& getTSNERoiAction() { return _tsneROIActions.getTSNERoiAction(); }
    TriggerAction& getTSNELandmarkAction() { return _tsneROIActions.getTSNELandmarkAction(); }

private:
    QMap<QString, utils::knn_library>      _knnAlgs;            /** Map of knn Alg names in GUI and knn enum values*/
    QMap<QString, QStringList>             _knnMetrics;         /** Map of knn Metric names in GUI and knn enum values*/

protected:
    HsneSettingsAction&     _hsneSettingsAction;                /** Reference to HSNE settings action */
    OptionAction            _knnTypeAction;                     /** KNN action */
    OptionAction            _distanceMetricAction;              /** Distance metric action */
    IntegralAction          _perplexityAction;                  /** Perplexity action */
    IntegralAction          _numScalesAction;                   /** Num scales action */
    IntegralAction          _seedAction;                        /** Random seed action */
    ToggleAction            _useMonteCarloSamplingAction;       /** Use Monte Carlo sampling on/off action */
    TriggerAction           _initAction;                        /** Init HSNE action */
    TsneROIActions          _tsneROIActions;                    /** tSNE ROI action */
};

