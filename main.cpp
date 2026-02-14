#include "nd_array.hpp"

#include <iomanip>
#include <iostream>
#include <vector>


void print_separator( const std::string& title )
{
	std::cout << "\n=== " << title << " ===\n";
}

int main( )
{
	std::vector<int> extents = { 3, 4, 5 };
	nd_array<int> arr3d_vec( extents );

	arr3d_vec.fill( 7 );

	print_separator( "3D array from vector (3x4x5) filled with 7" );
	std::cout << "3D Array from vector extents:\n";
	for( size_t i = 0; i < extents[0]; ++i )
	{
		std::cout << "Layer " << i << ":\n";
		for( size_t j = 0; j < extents[1]; ++j )
		{
			for( size_t k = 0; k < extents[2]; ++k )
			{
				std::cout << std::setw( 4 ) << arr3d_vec( i, j, k ) << " ";
			}
			std::cout << "\n";
		}
	}

	print_separator( "Creating 2D array (3x4)" );
	nd_array<double> arr2d( 3, 4 );

	// Fill with values
	for( size_t i = 0; i < 3; ++i )
	{
		for( size_t j = 0; j < 4; ++j )
		{
			arr2d( i, j ) = i * 4 + j;
		}
	}

	std::cout << "2D Array:\n";
	for( size_t i = 0; i < 3; ++i )
	{
		for( size_t j = 0; j < 4; ++j )
		{
			std::cout << std::setw( 6 ) << arr2d( i, j ) << " ";
		}
		std::cout << "\n";
	}

	print_separator( "Creating 3D array (2x3x4)" );
	nd_array<int> arr3d( 2, 3, 4 );

	// Fill with values
	int counter = 0;
	for( size_t i = 0; i < 2; ++i )
	{
		for( size_t j = 0; j < 3; ++j )
		{
			for( size_t k = 0; k < 4; ++k )
			{
				arr3d( i, j, k ) = counter++;
			}
		}
	}

	std::cout << "3D Array (layer by layer):\n";
	for( size_t i = 0; i < 2; ++i )
	{
		std::cout << "Layer " << i << ":\n";
		for( size_t j = 0; j < 3; ++j )
		{
			for( size_t k = 0; k < 4; ++k )
			{
				std::cout << std::setw( 4 ) << arr3d( i, j, k ) << " ";
			}
			std::cout << "\n";
		}
	}

	print_separator( "Array properties" );
	std::cout << "2D array rank: " << arr2d.rank( ) << "\n";
	std::cout << "2D array size: " << arr2d.size( ) << "\n";
	std::cout << "2D array extent(0): " << arr2d.extent( 0 ) << "\n";
	std::cout << "2D array extent(1): " << arr2d.extent( 1 ) << "\n";
	std::cout << "3D array rank: " << arr3d.rank( ) << "\n";
	std::cout << "3D array size: " << arr3d.size( ) << "\n";

	print_separator( "Subspan - getting a row from 2D array" );
	auto row1 = arr2d.subspan( 0, 1, 2 ); // Get row 1 (from index 1 to 2)
	std::cout << "Row 1 of 2D array: ";
	for( size_t j = 0; j < row1.extent( 1 ); ++j )
	{
		std::cout << row1( 0, j ) << " ";
	}
	std::cout << "\n";

	print_separator( "Subspan - getting a column range" );
	auto cols = arr2d.subspan( 1, 1, 3 ); // Get columns 1-2
	std::cout << "Columns 1-2 of 2D array:\n";
	for( size_t i = 0; i < cols.extent( 0 ); ++i )
	{
		for( size_t j = 0; j < cols.extent( 1 ); ++j )
		{
			std::cout << std::setw( 6 ) << cols( i, j ) << " ";
		}
		std::cout << "\n";
	}

	print_separator( "Slice - reducing dimension" );
	auto slice0 = arr3d.slice( 0, 1 ); // Get second layer (index 1)
	std::cout << "Slice of 3D array (layer 1):\n";
	std::cout << "Slice rank: " << slice0.rank( ) << "\n";
	for( size_t j = 0; j < slice0.extent( 0 ); ++j )
	{
		for( size_t k = 0; k < slice0.extent( 1 ); ++k )
		{
			std::cout << std::setw( 4 ) << slice0( j, k ) << " ";
		}
		std::cout << "\n";
	}

	print_separator( "Fill operation" );
	nd_array<int> arr2d_fill( 2, 3 );
	arr2d_fill.fill( 42 );
	std::cout << "Array filled with 42:\n";
	for( size_t i = 0; i < 2; ++i )
	{
		for( size_t j = 0; j < 3; ++j )
		{
			std::cout << arr2d_fill( i, j ) << " ";
		}
		std::cout << "\n";
	}

	print_separator( "Apply function" );
	arr2d_fill.apply( []( int x ) { return x * 2; } );
	std::cout << "Array after applying x*2:\n";
	for( size_t i = 0; i < 2; ++i )
	{
		for( size_t j = 0; j < 3; ++j )
		{
			std::cout << arr2d_fill( i, j ) << " ";
		}
		std::cout << "\n";
	}

	print_separator( "Copy constructor" );
	nd_array<int> arr_copy = arr2d_fill;
	std::cout << "Copied array:\n";
	for( size_t i = 0; i < 2; ++i )
	{
		for( size_t j = 0; j < 3; ++j )
		{
			std::cout << arr_copy( i, j ) << " ";
		}
		std::cout << "\n";
	}

	print_separator( "Modifying original after copy" );
	arr2d_fill.fill( 99 );
	std::cout << "Original array (filled with 99):\n";
	for( size_t i = 0; i < 2; ++i )
	{
		for( size_t j = 0; j < 3; ++j )
		{
			std::cout << arr2d_fill( i, j ) << " ";
		}
		std::cout << "\n";
	}
	std::cout << "Copied array (should still be 84):\n";
	for( size_t i = 0; i < 2; ++i )
	{
		for( size_t j = 0; j < 3; ++j )
		{
			std::cout << arr_copy( i, j ) << " ";
		}
		std::cout << "\n";
	}

	print_separator( "Dynamic rank array using initializer list" );
	nd_array<float> arr_dynamic( { 2, 3, 2 } );
	std::cout << "Dynamic rank: " << arr_dynamic.rank( ) << "\n";
	std::cout << "Dynamic size: " << arr_dynamic.size( ) << "\n";

	std::cout << "\n✓ All tests completed successfully!\n";

	return 0;
}
