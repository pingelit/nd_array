#include <catch2/catch_test_macros.hpp>
#include "../nd_array.hpp"
#include <vector>
#include <array>

TEST_CASE("nd_span - Construction", "[nd_span][construction]")
{
	SECTION("Variadic constructor - 1D")
	{
		int data[10] = {};
		nd_span<int> span(data, 10);
		REQUIRE(span.rank() == 1);
		REQUIRE(span.extent(0) == 10);
	}

	SECTION("Variadic constructor - 2D")
	{
		double data[12] = {};
		nd_span<double> span(data, 3, 4);
		REQUIRE(span.rank() == 2);
		REQUIRE(span.extent(0) == 3);
		REQUIRE(span.extent(1) == 4);
	}

	SECTION("Variadic constructor - 3D")
	{
		float data[24] = {};
		nd_span<float> span(data, 2, 3, 4);
		REQUIRE(span.rank() == 3);
		REQUIRE(span.extent(0) == 2);
		REQUIRE(span.extent(1) == 3);
		REQUIRE(span.extent(2) == 4);
	}

	SECTION("Initializer list constructor")
	{
		int data[24] = {};
		nd_span<int> span(data, {2, 3, 4});
		REQUIRE(span.rank() == 3);
		REQUIRE(span.extent(0) == 2);
		REQUIRE(span.extent(1) == 3);
		REQUIRE(span.extent(2) == 4);
	}

	SECTION("Container constructor - vector")
	{
		std::vector<int> data(24);
		std::vector<size_t> extents = {2, 3, 4};
		nd_span<int> span(data.data(), extents);
		REQUIRE(span.rank() == 3);
		REQUIRE(span.extent(0) == 2);
		REQUIRE(span.extent(1) == 3);
		REQUIRE(span.extent(2) == 4);
	}

	SECTION("Wrapping std::vector")
	{
		std::vector<int> vec = {1, 2, 3, 4, 5, 6};
		nd_span<int> span(vec.data(), 2, 3);
		REQUIRE(span.rank() == 2);
		REQUIRE(span(0, 0) == 1);
		REQUIRE(span(1, 2) == 6);
	}
}

TEST_CASE("nd_span - Element access", "[nd_span][access]")
{
	SECTION("1D span access")
	{
		int data[5] = {10, 20, 30, 40, 50};
		nd_span<int> span(data, 5);

		for(size_t i = 0; i < 5; ++i)
		{
			REQUIRE(span(i) == static_cast<int>((i + 1) * 10));
		}
	}

	SECTION("2D span access")
	{
		int data[12];
		for(size_t i = 0; i < 12; ++i)
		{
			data[i] = static_cast<int>(i);
		}

		nd_span<int> span(data, 3, 4);
		for(size_t i = 0; i < 3; ++i)
		{
			for(size_t j = 0; j < 4; ++j)
			{
				REQUIRE(span(i, j) == static_cast<int>(i * 4 + j));
			}
		}
	}

	SECTION("3D span access")
	{
		int data[24];
		for(size_t i = 0; i < 24; ++i)
		{
			data[i] = static_cast<int>(i);
		}

		nd_span<int> span(data, 2, 3, 4);
		int counter = 0;
		for(size_t i = 0; i < 2; ++i)
		{
			for(size_t j = 0; j < 3; ++j)
			{
				for(size_t k = 0; k < 4; ++k)
				{
					REQUIRE(span(i, j, k) == counter++);
				}
			}
		}
	}

	SECTION("Modifications through span affect underlying data")
	{
		int data[6] = {0, 0, 0, 0, 0, 0};
		nd_span<int> span(data, 2, 3);
		span(1, 2) = 99;

		REQUIRE(data[5] == 99); // Last element
	}

	SECTION("Out of bounds access throws")
	{
		int data[12] = {};
		nd_span<int> span(data, 3, 4);
		REQUIRE_THROWS_AS(span(3, 0), std::out_of_range);
		REQUIRE_THROWS_AS(span(0, 4), std::out_of_range);
	}
}

TEST_CASE("nd_span - Const access", "[nd_span][const]")
{
	SECTION("Const span access")
	{
		const int data[6] = {1, 2, 3, 4, 5, 6};
		nd_span<const int> span(data, 2, 3);

		REQUIRE(span(0, 0) == 1);
		REQUIRE(span(1, 2) == 6);
	}
}

TEST_CASE("nd_span - Subspan", "[nd_span][subspan]")
{
	SECTION("Subspan along dimension 0")
	{
		int data[20];
		for(size_t i = 0; i < 20; ++i)
		{
			data[i] = static_cast<int>(i);
		}

		nd_span<int> span(data, 4, 5);
		auto sub = span.subspan(0, 1, 3); // rows 1-2

		REQUIRE(sub.rank() == 2);
		REQUIRE(sub.extent(0) == 2);
		REQUIRE(sub.extent(1) == 5);
		REQUIRE(sub(0, 0) == 5);
		REQUIRE(sub(1, 0) == 10);
	}

	SECTION("Subspan along dimension 1")
	{
		int data[15];
		for(size_t i = 0; i < 15; ++i)
		{
			data[i] = static_cast<int>(i);
		}

		nd_span<int> span(data, 3, 5);
		auto sub = span.subspan(1, 1, 4); // cols 1-3

		REQUIRE(sub.rank() == 2);
		REQUIRE(sub.extent(0) == 3);
		REQUIRE(sub.extent(1) == 3);
		REQUIRE(sub(0, 0) == 1);
		REQUIRE(sub(0, 1) == 2);
	}

	SECTION("Subspan modifications affect original")
	{
		int data[12] = {};
		nd_span<int> span(data, 3, 4);

		auto sub = span.subspan(0, 1, 2);
		sub(0, 0) = 99;

		REQUIRE(data[4] == 99); // row 1, col 0
	}

	SECTION("Invalid subspan throws")
	{
		int data[12] = {};
		nd_span<int> span(data, 3, 4);
		REQUIRE_THROWS_AS(span.subspan(0, 2, 1), std::out_of_range); // start >= end
		REQUIRE_THROWS_AS(span.subspan(0, 0, 5), std::out_of_range); // end > extent
		REQUIRE_THROWS_AS(span.subspan(2, 0, 1), std::out_of_range); // dim >= rank
	}
}

TEST_CASE("nd_span - Slice", "[nd_span][slice]")
{
	SECTION("Slice 3D to 2D")
	{
		int data[24];
		for(size_t i = 0; i < 24; ++i)
		{
			data[i] = static_cast<int>(i);
		}

		nd_span<int> span(data, 2, 3, 4);
		auto slice = span.slice(0, 1); // Second layer

		REQUIRE(slice.rank() == 2);
		REQUIRE(slice.extent(0) == 3);
		REQUIRE(slice.extent(1) == 4);
		REQUIRE(slice(0, 0) == 12); // First element of second layer
	}

	SECTION("Slice 2D to 1D")
	{
		int data[12];
		for(size_t i = 0; i < 12; ++i)
		{
			data[i] = static_cast<int>(i);
		}

		nd_span<int> span(data, 3, 4);
		auto slice = span.slice(0, 1); // Second row

		REQUIRE(slice.rank() == 1);
		REQUIRE(slice.extent(0) == 4);
		REQUIRE(slice(0) == 4);
		REQUIRE(slice(1) == 5);
	}

	SECTION("Slice modifications affect original")
	{
		int data[60] = {};
		nd_span<int> span(data, 3, 4, 5);

		auto slice = span.slice(0, 1);
		slice(0, 0) = 99;

		REQUIRE(data[20] == 99); // layer 1, row 0, col 0
	}

	SECTION("Invalid slice throws")
	{
		int data[12] = {};
		nd_span<int> span(data, 3, 4);
		REQUIRE_THROWS_AS(span.slice(2, 0), std::out_of_range); // dim >= rank
		REQUIRE_THROWS_AS(span.slice(0, 3), std::out_of_range); // index >= extent
	}
}

TEST_CASE("nd_span - Properties", "[nd_span][properties]")
{
	SECTION("Rank and extents")
	{
		int data[24] = {};
		nd_span<int> span(data, 2, 3, 4);

		REQUIRE(span.rank() == 3);
		REQUIRE(span.extent(0) == 2);
		REQUIRE(span.extent(1) == 3);
		REQUIRE(span.extent(2) == 4);
		REQUIRE(span.max_rank() == 8);
	}

	SECTION("Data pointer")
	{
		int data[6] = {1, 2, 3, 4, 5, 6};
		nd_span<int> span(data, 2, 3);

		REQUIRE(span.data() == data);
		REQUIRE(span.data()[0] == 1);
	}

	SECTION("Invalid extent throws")
	{
		int data[6] = {};
		nd_span<int> span(data, 2, 3);
		REQUIRE_THROWS_AS(span.extent(2), std::out_of_range);
	}
}

TEST_CASE("nd_span - C-array interop", "[nd_span][c-interop]")
{
	SECTION("Wrapping C-array")
	{
		double* c_array = new double[12];
		for(size_t i = 0; i < 12; ++i)
		{
			c_array[i] = i * 1.5;
		}

		nd_span<double> span(c_array, 3, 4);
		REQUIRE(span.rank() == 2);
		REQUIRE(span(0, 0) == 0.0);
		REQUIRE(span(2, 3) == 11 * 1.5);

		delete[] c_array;
	}

	SECTION("Wrapping std::array")
	{
		std::array<int, 6> arr = {10, 20, 30, 40, 50, 60};
		nd_span<int> span(arr.data(), 2, 3);

		REQUIRE(span(0, 0) == 10);
		REQUIRE(span(1, 2) == 60);
	}
}

TEST_CASE("nd_span - Integration with nd_array", "[nd_span][integration]")
{
	SECTION("Creating span from nd_array")
	{
		nd_array<int> arr(3, 4);
		arr.fill(42);

		auto span = arr.subspan(0, 1, 3);
		REQUIRE(span.rank() == 2);
		REQUIRE(span(0, 0) == 42);
	}

	SECTION("Span modifications affect nd_array")
	{
		nd_array<int> arr(3, 4);
		arr.fill(0);

		auto span = arr.subspan(0, 1, 2);
		span(0, 0) = 99;

		REQUIRE(arr(1, 0) == 99);
	}
}
