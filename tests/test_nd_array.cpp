#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_template_test_macros.hpp>
#include "../nd_array.hpp"
#include <vector>

TEST_CASE("nd_array - Construction", "[nd_array][construction]")
{
	SECTION("Default constructor")
	{
		nd_array<int> arr;
		REQUIRE(arr.rank() == 0);
		REQUIRE(arr.size() == 0);
	}

	SECTION("Variadic constructor - 1D")
	{
		nd_array<int> arr(10);
		REQUIRE(arr.rank() == 1);
		REQUIRE(arr.size() == 10);
		REQUIRE(arr.extent(0) == 10);
	}

	SECTION("Variadic constructor - 2D")
	{
		nd_array<double> arr(3, 4);
		REQUIRE(arr.rank() == 2);
		REQUIRE(arr.size() == 12);
		REQUIRE(arr.extent(0) == 3);
		REQUIRE(arr.extent(1) == 4);
	}

	SECTION("Variadic constructor - 3D")
	{
		nd_array<float> arr(2, 3, 4);
		REQUIRE(arr.rank() == 3);
		REQUIRE(arr.size() == 24);
		REQUIRE(arr.extent(0) == 2);
		REQUIRE(arr.extent(1) == 3);
		REQUIRE(arr.extent(2) == 4);
	}

	SECTION("Initializer list constructor")
	{
		nd_array<int> arr({2, 3, 4});
		REQUIRE(arr.rank() == 3);
		REQUIRE(arr.size() == 24);
		REQUIRE(arr.extent(0) == 2);
		REQUIRE(arr.extent(1) == 3);
		REQUIRE(arr.extent(2) == 4);
	}

	SECTION("Container constructor - vector")
	{
		std::vector<size_t> extents = {2, 3, 4};
		nd_array<int> arr(extents);
		REQUIRE(arr.rank() == 3);
		REQUIRE(arr.size() == 24);
		REQUIRE(arr.extent(0) == 2);
		REQUIRE(arr.extent(1) == 3);
		REQUIRE(arr.extent(2) == 4);
	}
}

TEST_CASE("nd_array - Element access", "[nd_array][access]")
{
	SECTION("1D array access")
	{
		nd_array<int> arr(5);
		for(size_t i = 0; i < 5; ++i)
		{
			arr(i) = static_cast<int>(i * 10);
		}

		for(size_t i = 0; i < 5; ++i)
		{
			REQUIRE(arr(i) == static_cast<int>(i * 10));
		}
	}

	SECTION("2D array access")
	{
		nd_array<int> arr(3, 4);
		for(size_t i = 0; i < 3; ++i)
		{
			for(size_t j = 0; j < 4; ++j)
			{
				arr(i, j) = static_cast<int>(i * 4 + j);
			}
		}

		for(size_t i = 0; i < 3; ++i)
		{
			for(size_t j = 0; j < 4; ++j)
			{
				REQUIRE(arr(i, j) == static_cast<int>(i * 4 + j));
			}
		}
	}

	SECTION("3D array access")
	{
		nd_array<int> arr(2, 3, 4);
		int counter = 0;
		for(size_t i = 0; i < 2; ++i)
		{
			for(size_t j = 0; j < 3; ++j)
			{
				for(size_t k = 0; k < 4; ++k)
				{
					arr(i, j, k) = counter++;
				}
			}
		}

		counter = 0;
		for(size_t i = 0; i < 2; ++i)
		{
			for(size_t j = 0; j < 3; ++j)
			{
				for(size_t k = 0; k < 4; ++k)
				{
					REQUIRE(arr(i, j, k) == counter++);
				}
			}
		}
	}

	SECTION("Out of bounds access throws")
	{
		nd_array<int> arr(3, 4);
		REQUIRE_THROWS_AS(arr(3, 0), std::out_of_range);
		REQUIRE_THROWS_AS(arr(0, 4), std::out_of_range);
	}
}

TEST_CASE("nd_array - Copy semantics", "[nd_array][copy]")
{
	SECTION("Copy constructor")
	{
		nd_array<int> arr1(3, 4);
		arr1.fill(42);

		nd_array<int> arr2 = arr1;

		REQUIRE(arr2.rank() == arr1.rank());
		REQUIRE(arr2.size() == arr1.size());
		REQUIRE(arr2.extent(0) == arr1.extent(0));
		REQUIRE(arr2.extent(1) == arr1.extent(1));

		// Verify deep copy
		arr1.fill(99);
		REQUIRE(arr2(0, 0) == 42);
		REQUIRE(arr1(0, 0) == 99);
	}

	SECTION("Copy assignment")
	{
		nd_array<int> arr1(3, 4);
		arr1.fill(42);

		nd_array<int> arr2(2, 2);
		arr2 = arr1;

		REQUIRE(arr2.rank() == arr1.rank());
		REQUIRE(arr2.size() == arr1.size());

		// Verify deep copy
		arr1.fill(99);
		REQUIRE(arr2(0, 0) == 42);
	}
}

TEST_CASE("nd_array - Move semantics", "[nd_array][move]")
{
	SECTION("Move constructor")
	{
		nd_array<int> arr1(3, 4);
		arr1.fill(42);

		nd_array<int> arr2 = std::move(arr1);

		REQUIRE(arr2.rank() == 2);
		REQUIRE(arr2.size() == 12);
		REQUIRE(arr2(0, 0) == 42);
	}

	SECTION("Move assignment")
	{
		nd_array<int> arr1(3, 4);
		arr1.fill(42);

		nd_array<int> arr2;
		arr2 = std::move(arr1);

		REQUIRE(arr2.rank() == 2);
		REQUIRE(arr2.size() == 12);
		REQUIRE(arr2(0, 0) == 42);
	}
}

TEST_CASE("nd_array - Operations", "[nd_array][operations]")
{
	SECTION("Fill operation")
	{
		nd_array<int> arr(3, 4);
		arr.fill(42);

		for(size_t i = 0; i < 3; ++i)
		{
			for(size_t j = 0; j < 4; ++j)
			{
				REQUIRE(arr(i, j) == 42);
			}
		}
	}

	SECTION("Apply operation")
	{
		nd_array<int> arr(3, 4);
		arr.fill(10);
		arr.apply([](int x) { return x * 2; });

		for(size_t i = 0; i < 3; ++i)
		{
			for(size_t j = 0; j < 4; ++j)
			{
				REQUIRE(arr(i, j) == 20);
			}
		}
	}
}

TEST_CASE("nd_array - Subspan", "[nd_array][subspan]")
{
	SECTION("Subspan along dimension 0")
	{
		nd_array<int> arr(4, 5);
		for(size_t i = 0; i < 4; ++i)
		{
			for(size_t j = 0; j < 5; ++j)
			{
				arr(i, j) = static_cast<int>(i * 5 + j);
			}
		}

		auto sub = arr.subspan(0, 1, 3); // rows 1-2
		REQUIRE(sub.rank() == 2);
		REQUIRE(sub.extent(0) == 2);
		REQUIRE(sub.extent(1) == 5);
		REQUIRE(sub(0, 0) == 5);
		REQUIRE(sub(1, 0) == 10);
	}

	SECTION("Subspan along dimension 1")
	{
		nd_array<int> arr(3, 5);
		for(size_t i = 0; i < 3; ++i)
		{
			for(size_t j = 0; j < 5; ++j)
			{
				arr(i, j) = static_cast<int>(i * 5 + j);
			}
		}

		auto sub = arr.subspan(1, 1, 4); // cols 1-3
		REQUIRE(sub.rank() == 2);
		REQUIRE(sub.extent(0) == 3);
		REQUIRE(sub.extent(1) == 3);
		REQUIRE(sub(0, 0) == 1);
		REQUIRE(sub(0, 1) == 2);
	}

	SECTION("Subspan modifications affect original")
	{
		nd_array<int> arr(3, 4);
		arr.fill(0);

		auto sub = arr.subspan(0, 1, 2);
		sub(0, 0) = 99;

		REQUIRE(arr(1, 0) == 99);
	}

	SECTION("Invalid subspan throws")
	{
		nd_array<int> arr(3, 4);
		REQUIRE_THROWS_AS(arr.subspan(0, 2, 1), std::out_of_range); // start >= end
		REQUIRE_THROWS_AS(arr.subspan(0, 0, 5), std::out_of_range); // end > extent
		REQUIRE_THROWS_AS(arr.subspan(2, 0, 1), std::out_of_range); // dim >= rank
	}
}

TEST_CASE("nd_array - Slice", "[nd_array][slice]")
{
	SECTION("Slice 3D to 2D")
	{
		nd_array<int> arr(2, 3, 4);
		int counter = 0;
		for(size_t i = 0; i < 2; ++i)
		{
			for(size_t j = 0; j < 3; ++j)
			{
				for(size_t k = 0; k < 4; ++k)
				{
					arr(i, j, k) = counter++;
				}
			}
		}

		auto slice = arr.slice(0, 1); // Second layer
		REQUIRE(slice.rank() == 2);
		REQUIRE(slice.extent(0) == 3);
		REQUIRE(slice.extent(1) == 4);
		REQUIRE(slice(0, 0) == 12); // First element of second layer
	}

	SECTION("Slice 2D to 1D")
	{
		nd_array<int> arr(3, 4);
		for(size_t i = 0; i < 3; ++i)
		{
			for(size_t j = 0; j < 4; ++j)
			{
				arr(i, j) = static_cast<int>(i * 4 + j);
			}
		}

		auto slice = arr.slice(0, 1); // Second row
		REQUIRE(slice.rank() == 1);
		REQUIRE(slice.extent(0) == 4);
		REQUIRE(slice(0) == 4);
		REQUIRE(slice(1) == 5);
	}

	SECTION("Slice modifications affect original")
	{
		nd_array<int> arr(3, 4, 5);
		arr.fill(0);

		auto slice = arr.slice(0, 1);
		slice(0, 0) = 99;

		REQUIRE(arr(1, 0, 0) == 99);
	}
}

TEST_CASE("nd_array - Properties", "[nd_array][properties]")
{
	SECTION("Rank and size")
	{
		nd_array<int> arr(2, 3, 4);
		REQUIRE(arr.rank() == 3);
		REQUIRE(arr.size() == 24);
		REQUIRE(arr.max_rank() == 8);
	}

	SECTION("Extents")
	{
		nd_array<int> arr(2, 3, 4);
		REQUIRE(arr.extent(0) == 2);
		REQUIRE(arr.extent(1) == 3);
		REQUIRE(arr.extent(2) == 4);
	}

	SECTION("Data pointer")
	{
		nd_array<int> arr(2, 3);
		arr(0, 0) = 42;
		REQUIRE(arr.data()[0] == 42);
	}

	SECTION("Invalid extent throws")
	{
		nd_array<int> arr(2, 3);
		REQUIRE_THROWS_AS(arr.extent(2), std::out_of_range);
	}
}
