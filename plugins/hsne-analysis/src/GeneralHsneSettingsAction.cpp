#include "GeneralHsneSettingsAction.h"
#include "HsneSettingsAction.h"

#include "hdi/dimensionality_reduction/knn_utils.h"

#include <QHBoxLayout> 

using namespace mv::gui;

GeneralHsneSettingsAction::GeneralHsneSettingsAction(HsneSettingsAction& hsneSettingsAction) :
    GroupAction(&hsneSettingsAction, "GeneralHsneSettingsAction", true),
    _hsneSettingsAction(hsneSettingsAction),
    _knnTypeAction(this, "KNN Type"),
    _distanceMetricAction(this, "Distance metric"),
    _perplexityAction(this, "Perplexity"),
    _numScalesAction(this, "Hierarchy Scales"),
    _seedAction(this, "Random seed"),
    _useMonteCarloSamplingAction(this, "Use Monte Carlo sampling"),
    _initAction(this, "Init"),
    _tsneROIActions(this),
    _knnAlgs(),
    _knnMetrics()
{
    setText("HSNE");
    setObjectName("General HSNE");
    setLabelSizingType(LabelSizingType::Fixed);
    setLabelWidthFixed(100);

    /// UI set up: add actions
    for (auto& action : WidgetActions{ &_knnTypeAction, &_distanceMetricAction, &_perplexityAction,
        & _numScalesAction, & _seedAction, & _useMonteCarloSamplingAction, & _initAction, & _tsneROIActions})
        addAction(action);

    const auto& hsneParameters = hsneSettingsAction.getHsneParameters();

    _knnAlgs["HNSW"] = utils::knn_library::KNN_HNSW;
    _knnAlgs["ANNOY"] = utils::knn_library::KNN_ANNOY;
    _knnAlgs["Exact"] = utils::knn_library::KNN_EXACT;
    
    _knnMetrics["HNSW"] = { "Euclidean", "Inner Product (Dot)" };
    _knnMetrics["ANNOY"] = { "Euclidean", "Cosine", "Inner Product (Dot)", "Manhattan" };
    _knnMetrics["Exact"] = { "Euclidean" };

    _knnTypeAction.setDefaultWidgetFlags(OptionAction::ComboBox);
    _distanceMetricAction.setDefaultWidgetFlags(OptionAction::ComboBox);
    _perplexityAction.setDefaultWidgetFlags(IntegralAction::SpinBox | IntegralAction::Slider);
    _numScalesAction.setDefaultWidgetFlags(IntegralAction::SpinBox);
    _seedAction.setDefaultWidgetFlags(IntegralAction::SpinBox);
    _useMonteCarloSamplingAction.setDefaultWidgetFlags(ToggleAction::CheckBox);

    _knnTypeAction.initialize({ "ANNOY" , "HNSW", "Exact" }, "HNSW");
    _distanceMetricAction.initialize(_knnMetrics["ANNOY"], "Euclidean");
    _perplexityAction.initialize(2, 100, 30);
    _numScalesAction.initialize(1, 10, hsneParameters.getNumScales());
    _seedAction.initialize(-1000, 1000, hsneParameters.getSeed());
    _useMonteCarloSamplingAction.setChecked(hsneParameters.useMonteCarloSampling());
    
    _initAction.setToolTip("Initialize the HSNE hierarchy and create an embedding");
    _perplexityAction.setToolTip("Sets #nn to 3*perp + 1");

    const auto updateKnnAlgorithm = [this]() -> void {
         utils::knn_library knn_algo = _knnAlgs.value(_knnTypeAction.getCurrentText(), utils::knn_library::KNN_ANNOY);

         switch (knn_algo)
         {
         case utils::knn_library::KNN_HNSW:     _distanceMetricAction.setOptions(_knnMetrics["HNSW"]); break;
         case utils::knn_library::KNN_EXACT:    _distanceMetricAction.setOptions(_knnMetrics["Exact"]); break;
         case utils::knn_library::KNN_ANNOY:    // default
         default:                               _distanceMetricAction.setOptions(_knnMetrics["ANNOY"]);
         }

         // TODO: only one parameter setting container should be enough
        _hsneSettingsAction.getHsneParameters().setKnnLibrary(knn_algo);
        _hsneSettingsAction.getTsneParameters().setKnnAlgorithm(knn_algo);
    };

    const auto updateDistanceMetric = [this]() -> void {
        hdi::dr::knn_distance_metric knn_metric = hdi::dr::knn_distance_metric::KNN_METRIC_EUCLIDEAN;

        if (_distanceMetricAction.getCurrentText() == "Euclidean")
            knn_metric = hdi::dr::knn_distance_metric::KNN_METRIC_EUCLIDEAN;

        if (_distanceMetricAction.getCurrentText() == "Cosine")
            knn_metric = hdi::dr::knn_distance_metric::KNN_METRIC_COSINE;

        if (_distanceMetricAction.getCurrentText() == "Inner Product (Dot)")
            knn_metric = hdi::dr::knn_distance_metric::KNN_METRIC_INNER_PRODUCT;

        if (_distanceMetricAction.getCurrentText() == "Manhattan")
            knn_metric = hdi::dr::knn_distance_metric::KNN_METRIC_MANHATTAN;

        _hsneSettingsAction.getHsneParameters().setAknnMetric(knn_metric);
        _hsneSettingsAction.getTsneParameters().setKnnDistanceMetric(knn_metric);
    };

    const auto updatePerplexity = [this]() -> void {
        _hsneSettingsAction.getHsneParameters().setNNWithPerplexity(_perplexityAction.getValue());
        _hsneSettingsAction.getTsneParameters().setPerplexity(_perplexityAction.getValue());
    };

    const auto updateNumScales = [this]() -> void {
        _hsneSettingsAction.getHsneParameters().setNumScales(_numScalesAction.getValue());
        _hsneSettingsAction.getInteractiveScaleAction().getScaleDownUpActions().setNumScales(_numScalesAction.getValue());
    };

    const auto updateSeed = [this]() -> void {
        _hsneSettingsAction.getHsneParameters().setSeed(_seedAction.getValue());
    };

    const auto updateUseMonteCarloSampling = [this]() -> void {
        _hsneSettingsAction.getHsneParameters().useMonteCarloSampling(_useMonteCarloSamplingAction.isChecked());
    };

    const auto updateReadOnly = [this]() -> void {
        const auto enabled = !isReadOnly();

        _initAction.setEnabled(enabled);
        _tsneROIActions.setEnabled(enabled);
        _knnTypeAction.setEnabled(enabled);
        _distanceMetricAction.setEnabled(enabled);
        _perplexityAction.setEnabled(enabled);
        _numScalesAction.setEnabled(enabled);
        _seedAction.setEnabled(enabled);
        _useMonteCarloSamplingAction.setEnabled(enabled);
    };

    connect(&_knnTypeAction, &OptionAction::currentIndexChanged, this, [this, updateKnnAlgorithm]() {
        updateKnnAlgorithm();
    });

    connect(&_distanceMetricAction, &OptionAction::currentIndexChanged, this, [this, updateDistanceMetric](const std::int32_t& currentIndex) {
        updateDistanceMetric();
    });

    connect(&_perplexityAction, &IntegralAction::valueChanged, this, [this, updatePerplexity](const std::int32_t& value) {
        updatePerplexity();
    });

    connect(&_numScalesAction, &IntegralAction::valueChanged, this, [this, updateNumScales]() {
        updateNumScales();
    });

    connect(&_seedAction, &IntegralAction::valueChanged, this, [this, updateSeed]() {
        updateSeed();
    });

    connect(&_useMonteCarloSamplingAction, &ToggleAction::toggled, this, [this, updateUseMonteCarloSampling]() {
        updateUseMonteCarloSampling();
    });

    connect(&_initAction, &ToggleAction::toggled, this, [this](bool toggled) {
        setReadOnly(toggled);
    });

    connect(this, &GroupAction::readOnlyChanged, this, [this, updateReadOnly](const bool& readOnly) {
        updateReadOnly();
    });

    updateKnnAlgorithm();
    updateDistanceMetric();
    updatePerplexity();
    updateNumScales();
    updateSeed();
    updateUseMonteCarloSampling();
    updateReadOnly();
}

/// ////////////// ///
/// TsneROIActions ///
/// ////////////// ///


TsneROIActions::TsneROIActions(QObject* parent) :
    WidgetAction(parent, "TsneROIActions"),
    _tsneROIAction(this, "ROI"),
    _tsneLandmarkROIAction(this, "Landmarks")
{
    setText("t-SNE");

    _tsneROIAction.setToolTip("Compute a t-SNE for all data points in the current ROI");
    _tsneLandmarkROIAction.setToolTip("Compute a t-SNE for all current landmarks");
}


TsneROIActions::Widget::Widget(QWidget* parent, TsneROIActions* tsneROIActions) :
    WidgetActionWidget(parent, tsneROIActions)
{
    auto layout = new QHBoxLayout();

    layout->setContentsMargins(0, 0, 0, 0);

    layout->addWidget(tsneROIActions->getTSNERoiAction().createWidget(this));
    layout->addWidget(tsneROIActions->getTSNELandmarkAction().createWidget(this));

    setLayout(layout);
}

