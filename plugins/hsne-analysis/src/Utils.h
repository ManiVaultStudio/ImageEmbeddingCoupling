#ifndef UTILS_H
#define UTILS_H

#include <cmath>        // sqrt, sin, cos
#include <numeric>      // accumulate
#include <algorithm>    // for_each, max
#include <execution>
#include <vector>
#include <random>       // random_device, mt19937, uniform_real_distribution
#include <chrono>       // high_resolution_clock, milliseconds
#include <string>
#include <iostream>
#include <iterator>     // std::forward_iterator_tag
#include <cstddef>      // std::ptrdiff_t
#include <type_traits>
#include <typeinfo>
#include <functional>

#include "graphics/Vector2f.h"  // mv::Vector2f

#include "CommonTypes.h"
#include "Logger.h"

namespace hdi {
    namespace dr {
        enum knn_library;
    }
}

namespace utils {

    /// /////// ///
    /// Looping ///
    /// /////// ///

    // Motivated by 
    // https://github.com/klmr/cpp11-range
    // https://isocpp.org/blog/2020/12/writing-a-custom-iterator-in-modern-cpp
    // cpp11-range does not define a forward_iterator but only a input_iterator
    // and can therefor not be used with std::execution in e.g. std::for_each
    // Other resources: https://github.com/VinGarcia/Simple-Iterator-Template

    template <typename T>
    struct pyrange {
        static_assert(std::is_integral<T>::value, "Integral type required.");

        // see https://en.cppreference.com/w/cpp/iterator/iterator_traits
        struct pyrange_iter
        {
            using iterator_category = std::forward_iterator_tag;    // c++17 style, c++20 would use std::forward_iterator
            using difference_type = std::ptrdiff_t;
            using value_type = T;
            using pointer = value_type*;
            using reference = value_type&;

            pyrange_iter() : _val(0) {}
            pyrange_iter(value_type val) : _val(val) {}

            reference operator*() { return _val; }
            pointer operator->() { return &_val; }

            pyrange_iter& operator++() { _val++; return *this; }                                // prefix increment
            pyrange_iter  operator++(int) { pyrange_iter tmp = *this; ++(*this); return tmp; }  // postix increment

            friend bool operator== (const pyrange_iter& a, const pyrange_iter& b) { return a._val == b._val; };
            friend bool operator!= (const pyrange_iter& a, const pyrange_iter& b) { return a._val != b._val; };

        private:
            value_type _val;
        };

        pyrange(T end) : _begin(0), _end(end) { }
        // if end < begin, don't throw an exception but instead don't iterate at all
        pyrange(T begin, T end) : _begin(begin), _end(end) { if (end < begin) _end = _begin; }

        pyrange_iter begin() { return _begin; }
        pyrange_iter end() { return _end; }

        const char* typeinfo() const { return typeid(T).name(); }

    private:
        pyrange_iter _begin;
        pyrange_iter _end;

    };


    /*! Use parallel loops in release and sequenced loops in debug mode. 
    *   Creates a sequence of n numbers, e.g. for  n=3 -> 0,1,2
    *   Use as:
    *
        auto range = utils::pyrange(n);
        std::for_each(utils::exec_policy, range.begin(), range.end(), [&](const auto i) {
            myFunction(i);
        });
    *
    */
#ifdef NDEBUG
    static auto exec_policy = std::execution::par_unseq;
#else
    static auto exec_policy = std::execution::seq;
#endif

    /// ///// ///
    /// ENUMS ///
    /// ///// ///

    // extended HDILib hdi::dr::knn_library
    enum class knn_library
    {
        KNN_HNSW = 0,   // same as in HDILib
        KNN_ANNOY = 1,  // same as in HDILib
        KNN_FAISS = 2,  // not implemented
        KNN_EXACT = 3
    };

    bool convertToHDILibKnnLib(const utils::knn_library& in, hdi::dr::knn_library& out);

    enum class TraversalDirection
    {
        UP,
        DOWN,
        AUTO
    };

    // No checks here
    inline void applyTraversalDirection(const utils::TraversalDirection& direction, uint32_t& scaleLevel)
    {
        if (direction == utils::TraversalDirection::UP)
            scaleLevel++;
        if (direction == utils::TraversalDirection::DOWN)
            scaleLevel--;
    }

    /// //// ///
    /// MATH ///
    /// //// ///

    // interpolate three 2d points
    inline mv::Vector2f interpol2D(const mv::Vector2f& vec1, const mv::Vector2f& vec2, const mv::Vector2f& vec3) {
        return { /* x = */ (vec1.x + vec2.x + vec3.x) / 3.0f,
                 /* y = */ (vec1.y + vec2.y + vec3.y) / 3.0f };
    }

    // extendX and extendY are absolute values
    inline mv::Vector2f randomVec(const float radiusX, const float radiusY) {
        static std::random_device rd;  // Will be used to obtain a seed for the random number engine
        static std::mt19937 gen(rd()); // Standard mersenne_twister_engine seeded with rd()
        static std::uniform_real_distribution<float> dis(0, 1);

        const float maxR = std::max(radiusX, radiusY);  // sample from a circle - usually radiusX and radiusY are similar

        assert(maxR >= 0);

        const float r = maxR * std::sqrt(dis(gen));     // random radius: uniformly sample from [0, 1], sqrt (important!), then scale to [0, maxR]
        const float t = 2.0f * 3.141592f * dis(gen);    // random angle: uniformly sample from [0, 1] and scale to [0, pi]

        return { /* x = */ r * std::cos(t), 
                 /* y = */ r * std::sin(t) };
    }

    // Cyclic group of order "size"
    // https://godbolt.org/z/nKoc785Ga
    /* Example
        CyclicGroup a(0, 3);

        auto b = a + 1;     // 1
        auto c = a + 2;     // 2
        auto d = a + 3;     // 0
        auto e = a + 4;     // 1
    */
    class CyclicGroup
    {
    public:
        constexpr CyclicGroup() : _size(0), _value(0) {}
        constexpr CyclicGroup(size_t size) : _size(size), _value(0) {}
        constexpr CyclicGroup(size_t val, size_t size) : _size(size), _value(val) {}

        constexpr size_t value() const { return _value; }
        constexpr size_t size() const { return _size; }

        constexpr void setValue(size_t num) { _value = mod(num); }

        // Overloading the prefix operator
        constexpr CyclicGroup& operator++() {
            _value = mod(_value + 1);
            return *this;
        }

        // Overloading the postfix operator
        constexpr CyclicGroup operator++(int) {
            CyclicGroup temp(_value, _size);
            ++(*this);
            return temp;
        }

        constexpr CyclicGroup operator+(CyclicGroup const& rhs) const {
            return CyclicGroup(mod(_value + rhs.value()), size());
        }

        constexpr CyclicGroup operator+(int64_t rhs) const {
            return CyclicGroup(mod(_value + rhs), size());
        }

        constexpr bool operator==(const CyclicGroup& rhs) const
        {
            return _value == rhs.value() && _size == rhs.size();
        }

        constexpr bool operator!=(const CyclicGroup& rhs) const
        {
            return _value != rhs.value() || _size != rhs.size();
        }

    private:

        constexpr size_t mod(size_t num) const { return num % _size; }

        size_t _value;
        size_t _size;
    };


    inline float sign(const mv::Vector2f& p1, const mv::Vector2f& p2, const mv::Vector2f& p3)
    {
        return (p1.x - p3.x) * (p2.y - p3.y) - (p2.x - p3.x) * (p1.y - p3.y);
    }

    // https://stackoverflow.com/a/2049593/16767931
    inline bool pointInTriangle(const mv::Vector2f& pt, const mv::Vector2f& v1, const mv::Vector2f& v2, const mv::Vector2f& v3)
    {
        float d1, d2, d3;
        bool has_neg, has_pos;

        d1 = utils::sign(pt, v1, v2);
        d2 = utils::sign(pt, v2, v3);
        d3 = utils::sign(pt, v3, v1);

        has_neg = (d1 < 0) || (d2 < 0) || (d3 < 0);
        has_pos = (d1 > 0) || (d2 > 0) || (d3 > 0);

        return !(has_neg && has_pos);
    }

    /*! Calculate the mean value for each channel
     *
     * \param numPoints number of data points
     * \param numDims number of dimensions
     * \param attribute_data data
     * \return vector with [mean_Ch0, mean_Ch1, ...]
     */
    template<typename T>
    std::vector<float> CalcMeanPerChannel(size_t numPoints, size_t numDims, const std::vector<T>& attribute_data) {
        std::vector<float> meanVals(numDims, 0);

        auto range = utils::pyrange(numDims);
        std::for_each(utils::exec_policy, range.begin(), range.end(), [&](const auto dimCount) {
            float sum = 0.0f;
            for (uint32_t pointCount = 0; pointCount < numPoints; pointCount++) {
                sum += attribute_data[pointCount * numDims + dimCount];
            }

            meanVals[dimCount] = sum / numPoints;
        });

        return meanVals;
    }

    /*! Calculate the mean value for each channel
     *
     * \param numPoints number of data points
     * \param numDims number of dimensions
     * \param attribute_data data
     * \return vector with [mean_Ch0, mean_Ch1, ...]
     */
    template<typename T>
    void CenterDataChannelwise(const size_t numPoints, const size_t numDims, const std::vector<T>& attribute_data, std::vector<T>& normed_data) {
        normed_data.resize(numPoints * numDims);

        std::vector<float> channelMeans = CalcMeanPerChannel(numPoints, numDims, attribute_data);

        auto range = utils::pyrange(numDims);
        std::for_each(utils::exec_policy, range.begin(), range.end(), [&](const auto dimCount) {
            for (uint32_t pointCount = 0; pointCount < numPoints; pointCount++) {
                normed_data[pointCount * numDims + dimCount] = attribute_data[pointCount * numDims + dimCount] - channelMeans[dimCount];
            }
        });
    }


    /// ////// ///
    /// TIMING ///
    /// ////// ///

    /* Logs the time of a lambda function, call like:
    * 
        utils::timer([&]() {
             <CODE YOU WANT TO TIME>
            },
            "<DESCRIPTION>");
    */
    template <typename F>
    void timer(F myFunc, std::string name) {
        using clock = std::chrono::high_resolution_clock;
        const auto time_start = clock::now();
        myFunc();
        Log::info("Timing " + name + ": " + std::to_string(std::chrono::duration_cast<std::chrono::milliseconds>(clock::now() - time_start).count()) + "ms");
    }

    /* Logs the time of a scope, call like:
    *
        <CODE>
        {
            utils::ScopedTimer myTimer("Scope workload");
            <SCOPED CODE YOU WANT TO TIME>
        }
        <CODE>
    */
    class ScopedTimer {
        using clock = std::chrono::high_resolution_clock;
    public:
        //! start the timer
        ScopedTimer(std::string title, std::function<void(std::string)> _logFunc = Log::info);
        //! stop the timer and save the elapsedTime
        ~ScopedTimer();

    private:
        std::chrono::time_point<clock> _start;
        std::string _title;
        std::function<void(std::string)> _logFunc;
    };


    /// //////////// ///
    /// DATA STRUCTS ///
    /// //////////// ///
    
    // Bi-directional map that is a bit more memory efficient than std::map<T1, T2> map1; std::map<T2, T1> map2;
    // Call like:
    /*
    UnorderedBimap<uint32_t, uint32_t> bimap;

    bimap.insert(3, 4);
    bimap.insert(5, 1);

    const auto search = bimap.findInAtoB(3);
    if (search != bimap.AtoBEnd()) {
        std::cout << "Found " << search->first << " " << bimap.value(search) << '\n';
    } else {
        std::cout << "Not found\n";
    }

    auto search2 = bimap.findInBtoA(1);
    if (search2 != bimap.BtoAEnd()) {
        std::cout << "Found " << search2->first << " " << bimap.value(search2) << '\n';
    } else {
        std::cout << "Not found\n";
    }

    > Found 3 4
    > Found 1 5
    */

    template <typename TA, typename TB>
    class UnorderedBimap {
        typedef std::unordered_map<TA, TB> MapAToB;
        typedef std::unordered_map<TB, typename MapAToB::iterator> MapBToA;

    public:

        UnorderedBimap() {
            _mapAtoB.max_load_factor(1);
            _mapBtoA.max_load_factor(1);
        }

        void insert(const TA a, const TB b) {
            _mapAtoB[a] = b;
            _mapBtoA[b] = _mapAtoB.find(a);
        }

        const auto findInAtoB(const TA key) const {
            return _mapAtoB.find(key);
        }

        const auto findInBtoA(const TB key) const {
            return _mapBtoA.find(key);
        }

        const auto AtoBEnd() const { return _mapAtoB.cend(); }
        const auto BtoAEnd() const { return _mapBtoA.cend(); }

        void clear() {
            _mapAtoB.clear();
            _mapBtoA.clear();
        }

        void reserve(const size_t capacity) {
            _mapAtoB.reserve(capacity);
            _mapBtoA.reserve(capacity);
        }

        TA value(const typename MapBToA::const_iterator key_it) const { return key_it->second->first; }
        TB value(const typename MapAToB::const_iterator key_it) const { return key_it->second; }

    private:

        MapAToB _mapAtoB;
        MapBToA _mapBtoA;
    };

    template<typename T>
    void eraseElements(std::vector<T>& container, const std::vector<uint32_t>& positionsToErase) {
        size_t currPos{ 0 };
        size_t nextIndPos{ 0 };
        container.erase(std::remove_if(container.begin(), container.end(), [&currPos, &nextIndPos, &positionsToErase](T)
            {
                if (nextIndPos < positionsToErase.size() && currPos++ == positionsToErase[nextIndPos])
                {
                    ++nextIndPos;
                    return true;
                }

                return false;
            }), container.end());
    }

    /// ////////// ///
    /// EMBEDDINGS ///
    /// ////////// ///
    class EmbeddingExtends
    {
    public:
        EmbeddingExtends();
        EmbeddingExtends(float x_min, float x_max, float y_min, float y_max);   // be sure that x_max >= x_min and y_max >= y_min

        void setExtends(float x_min, float x_max, float y_min, float y_max);    // be sure that x_max >= x_min and y_max >= y_min

        float x_min() const { return _x_min; }
        float x_max() const { return _x_max; }
        float y_min() const { return _y_min; }
        float y_max() const { return _y_max; }
        float extend_x() const { return _extend_x; }
        float extend_y() const { return _extend_y; }

        std::string getMinMaxString() const {
            return fmt::format("x in [{0}, {1}], y in [{2}, {3}]", _x_min, _x_max, _y_min, _y_max);
        }

        friend std::ostream& operator<<(std::ostream& os, const EmbeddingExtends& ext) {
            return os << ext.getMinMaxString() << std::endl;
        }

        friend std::string operator<<(const std::string& os, const EmbeddingExtends& ext) {
            return os + ext.getMinMaxString();
        }

    private:
        float _x_min, _x_max, _y_min, _y_max;
        float _extend_x, _extend_y;
    };


    EmbeddingExtends computeExtends(const std::vector<float>& emb);

    EmbeddingExtends computeExtends(const std::vector<mv::Vector2f>& emb);


    /// ///// ///
    /// IMAGE ///
    /// ///// ///
    class Vector2D {
    public:
        constexpr Vector2D() : _x(-1.0f), _y(-1.0f) {}
        Vector2D(float x, float y) : _x(x), _y(y) {}
        Vector2D(const Vector2D& vec) : _x(vec.x()), _y(vec.y()) {}

        float x() const { return _x; }
        float y() const { return _y; }

        void setX(float x) { _x = x; }
        void setY(float y) { _y = y; }

        friend bool operator==(Vector2D const& lhs, Vector2D const& rhs) {
            return (lhs.x() == rhs.x()) && (lhs.y() == rhs.y());
        }

        friend bool operator!=(Vector2D const& lhs, Vector2D const& rhs) {
            return (lhs.x() != rhs.x()) || (lhs.y() != rhs.y());
        }

    private:
        float _x;    // width
        float _y;    // height
    };
    
    struct ROI {
        constexpr ROI() : layerBottomLeft(), layerTopRight(), viewRoiXY(), viewRoiWH() {}
        ROI(uint32_t bottomLeftX, uint32_t bottomLeftY, uint32_t topRightX, uint32_t topRightY) :
            layerBottomLeft(static_cast<float>(bottomLeftX), static_cast<float>(bottomLeftY)), 
            layerTopRight(static_cast<float>(topRightX), static_cast<float>(topRightY)), 
            viewRoiXY(), 
            viewRoiWH() 
        {}

        ROI(Vector2D layerBottomLeft, Vector2D layerTopRight) : 
            layerBottomLeft(layerBottomLeft), layerTopRight(layerTopRight), viewRoiXY(), viewRoiWH() 
        {}

        ROI(uint32_t bottomLeftX, uint32_t bottomLeftY, uint32_t topRightX, uint32_t topRightY, float viewBottomLeftX, float viewBottomLeftY, float viewTopRightX, float viewTopRightY) :
            layerBottomLeft(static_cast<float>(bottomLeftX), static_cast<float>(bottomLeftY)),
            layerTopRight(static_cast<float>(topRightX), static_cast<float>(topRightY)),
            viewRoiXY(viewBottomLeftX, viewBottomLeftY),
            viewRoiWH(viewTopRightX, viewTopRightY) 
        {}

        ROI(Vector2D layerBottomLeft, Vector2D layerTopRight, Vector2D viewRoiXY, Vector2D viewRoiWH) : 
            layerBottomLeft(layerBottomLeft), layerTopRight(layerTopRight), viewRoiXY(viewRoiXY), viewRoiWH(viewRoiWH) {}

        // Layer ROI (image coordinates)
        Vector2D  layerBottomLeft;  /** ROI bottom left, .x() is width and .y() is height. (0,0) is buttom left from user perspective, x-axis goes to the right */
        Vector2D  layerTopRight;    /** ROI top right left, .x() is width and .y() is height. (0,0) is buttom left from user perspective, x-axis goes to the right */

        // View ROI (depends on viewer windows size)
        Vector2D  viewRoiXY;  /** ROI bottom left, .x() is width and .y() is height. (0,0) is buttom left from user perspective, x-axis goes to the right */
        Vector2D  viewRoiWH;    /** ROI top right left, .x() is width and .y() is height. (0,0) is buttom left from user perspective, x-axis goes to the right */

        static size_t computeNumPixelInROI(const Vector2D& layerBottomLeft, const Vector2D& layerTopRight)
        {
            constexpr auto uninitVec = Vector2D();
            if (layerBottomLeft == uninitVec && layerTopRight == uninitVec)
                return 0;

            size_t numRows = static_cast<size_t>(layerTopRight.x()) - static_cast<size_t>(layerBottomLeft.x());
            size_t numCols = static_cast<size_t>(layerTopRight.y()) - static_cast<size_t>(layerBottomLeft.y());

            return numRows * numCols;
        }

        size_t getNumPixelInROI() const
        {
            return computeNumPixelInROI(layerBottomLeft, layerTopRight);
        }

        friend bool operator==(ROI const& lhs, ROI const& rhs) {
            return (lhs.layerBottomLeft == rhs.layerBottomLeft) &&
                   (lhs.layerTopRight == rhs.layerTopRight) &&
                   (lhs.viewRoiXY == rhs.viewRoiXY) &&
                   (lhs.viewRoiWH == rhs.viewRoiWH);
        }
        
        friend bool operator!=(ROI const& lhs, ROI const& rhs) {
            return (lhs.layerBottomLeft != rhs.layerBottomLeft) ||
                (lhs.layerTopRight != rhs.layerTopRight) ||
                (lhs.viewRoiXY != rhs.viewRoiXY) ||
                (lhs.viewRoiWH != rhs.viewRoiWH);
        }

    };

    // check if (x,y) in in ROI
    inline bool pixelInRoi(const uint32_t x, const uint32_t y, const ROI& roi)
    {
        if (x == std::clamp(x, static_cast<uint32_t>(roi.layerBottomLeft.x()), static_cast<uint32_t>(roi.layerTopRight.x())) &&
            y == std::clamp(y, static_cast<uint32_t>(roi.layerBottomLeft.y()), static_cast<uint32_t>(roi.layerTopRight.y())))
            return true;

        return false;
    }

    /// /////////// ///
    /// INTERACTION ///
    /// /////////// ///

    class VisualBudgetRange {
    public:
        VisualBudgetRange() : _min(0), _max(1), _range(1), _target(1), _heuristic(false) {}
        VisualBudgetRange(size_t min, size_t max, size_t range, size_t target, bool heuristic) : _min(min), _max(max), _range(range), _target(target), _heuristic(heuristic) {}
    
        /** Checks wheter value is in [min, max] visual range
        * 1 in [1, 1]
          2 in [1, 4]
          4 in [1, 4]
          5 NOT in [1, 4]
          1 in [1, 4]
          0 NOT in [1, 4]
        */
        bool isWithinRange(size_t val) const {
            return val == std::clamp(val, _min, _max);
        }

        // Setter
        void setMin(const size_t newMin) {
            _min = newMin;

            if (_min + _range > _max)
                _max = _min + _range;
        }

        void setMax(const size_t newMax) {
            _max = newMax;

            if (_max - _range < _min)
                _min = _max - _range;
        }

        void setRange(const size_t newRange) {
            _range = newRange;

            if (_max - _min < _range)
                _max = _min + _range;

        }

        void setTarget(const size_t newTarget) { _target = newTarget; }

        void setHeuristic(const bool newHeuristic) { _heuristic = newHeuristic; }

        // Getter
        size_t getMin() const { return _min; };
        size_t getMax() const { return _max; };
        size_t getRange() const { return _range; };
        size_t getTarget() const { return _target; };
        bool getHeuristic() const { return _heuristic; };

    private:
        size_t _min;      //
        size_t _max;      //
        size_t _range;    //
        size_t _target;   //
        bool _heuristic;  //
    };

    class VisualTarget {
    public:
        VisualTarget() : _target(1), _heuristic(false) {}
        VisualTarget(size_t target, bool heuristic) : _target(target), _heuristic(heuristic) {}
        VisualTarget(const VisualBudgetRange& visualBudgetRange) : _target(visualBudgetRange.getTarget()), _heuristic(visualBudgetRange.getHeuristic()) {}
    
        // Setter
        void setTarget(const size_t newTarget) { _target = newTarget; }
        void setHeuristic(const bool newHeuristic) { _heuristic = newHeuristic; }

        // Getter
        size_t getTarget() const { return _target; };
        bool getHeuristic() const { return _heuristic; };

    private:
        size_t _target;   //
        bool _heuristic;  //
    };


    /// ///// ///
    /// LOCKS ///
    /// ///// ///

    class Lock
    {

    public:
        virtual bool isLocked() const = 0;
        virtual void reset() = 0;
        virtual void lock() = 0;
    };

    class BoolLock : Lock
    {

    public:
        BoolLock() : _value(false) {};
        BoolLock(bool state) : _value(state) {};

        bool isLocked() const {
            return _value;
        }

        void reset() { _value = false; };

        void lock() { _value = true; };

        void toggle() {
            _value = !_value;
        };

    private:
        bool _value;
    };

    class CyclicLock : Lock
    {

    public:
        CyclicLock() : _cylicGroup(0, 2) {};
        CyclicLock(size_t size) : _cylicGroup(0, size) {};
        CyclicLock(size_t value, size_t size) : _cylicGroup(value, size) {};

        static bool isLocked(const CyclicLock& lock)
        {
            return lock.isLocked();
        }

        bool isLocked() const {
            return _cylicGroup.value() > 0;
        }

        void reset() {
            _cylicGroup.setValue(0);
        };

        void lock() { 
            _cylicGroup.setValue(0);
        };

        void setValue(size_t val) {
            _cylicGroup.setValue(val);
        };

        // prefix
        CyclicLock& operator++() {
            _cylicGroup++;
            return *this;
        }

        // postfix
        CyclicLock operator++(int) {
            CyclicLock temp(_cylicGroup.value(), _cylicGroup.size());
            ++_cylicGroup;
            return temp;
        }

    private:
        CyclicGroup _cylicGroup;
    };

    template<class lockClass>
    class Locks
    {
    public:
        Locks() {};

        void addLock(std::string lockName) { _locks[lockName] = lockClass(); };

        void resetAll() {
            for (lockClass& lock : _locks)
                lock.reset();
        }

        lockClass& operator[](std::string lockName) {
            return _locks[lockName];
        }

        void visit(std::function<void(std::string, lockClass&)>const& func) {
            for (std::pair<const std::string, lockClass>& lockPair : _locks) {
                std::string name = lockPair.first;
                lockClass& lock = lockPair.second;
                func(name, lock);
            }
        }

    private:
        std::unordered_map<std::string, lockClass> _locks;
    };

}

#endif UTILS_H