#include <catch2/catch_test_macros.hpp>

#include "Utils.h"

#include <vector>

#include <graphics/Vector2f.h>  // mv::Vector2f


static inline bool equalVectors(const mv::Vector2f& a, const mv::Vector2f& b) {
	return (a - b).sqrMagnitude() < 0.000001f;
}

TEST_CASE("2D vector interpolation", "[math]")
{

	const mv::Vector2f inter = utils::interpol2D({ 1.0f, 0.0f }, { -1.0f, -0.0f }, { 0.0f, 3.0f });
	const mv::Vector2f expexted = { 0.0f, 1.0f };

	REQUIRE( equalVectors(inter, expexted) );

	/* Values created with MATLB
	
		x = -10 + (10+10)*rand(3,1); % x-coordinate
		y = -10 + (10+10)*rand(3,1); % y-coordinate
		x_centroid = mean(x);
		y_centroid = mean(y);
	*/
	const std::vector<mv::Vector2f> points{ {0.814723686393179f, 0.913375856139019f}, {0.905791937075619f, 0.632359246225410f}, {0.126986816293506f, 0.0975404049994095f},
		{-4.43003562265903f, 9.29777070398553f}, {0.937630384099677f,-6.84773836644903f}, {9.15013670868595f, 9.41185563521231f} ,
		{9.14333896485891f, -7.16227322745569f}, {-0.292487025543176f, -1.56477434747450f}, {6.00560937777600f, 8.31471050378134f} ,
		{5.84414659119109f, -9.28576642851621f}, {9.18984852785806f, 6.98258611737554f}, {3.11481398313174f, 8.67986495515101f}
	};

	const std::vector<mv::Vector2f> expected{ {0.615834146587435f, 0.547758502454613f},
		{1.885910490042199f, 3.953962657582936f},
		{4.952153772363913f, -0.137445690382950f} ,
		{6.049603034060294f, 2.125561548003449f}
	};

	for (size_t i = 0; i < expected.size(); i++) {
		REQUIRE(equalVectors(utils::interpol2D(points[3*i], points[3 * i + 1], points[3 * i + 2]), expected[i]));
	}
}