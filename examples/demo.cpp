#include "nd_array/nd_array.hpp"

#include <iomanip>
#include <iostream>
#include <vector>

using namespace cppa;

/// \brief Prints a section header to visually separate demo blocks.
/// \param title Title to display in the header.
void print_separator( const std::string& title )
{
	std::cout << "\n=== " << title << " ===\n";
}

/// \brief Simulates a C API that returns a raw heap-allocated array.
/// \param size Output size of the allocated array.
/// \return Pointer to the allocated array. Caller must delete[] it.
double* get_c_array_from_api( size_t& size )
{
	size        = 12;
	double* arr = new double[size];
	for( size_t i = 0; i < size; ++i )
	{
		arr[i] = i * 1.5;
	}
	return arr;
}

/// \brief Demonstrates wrapping a C API array with nd_span.
void demo_c_api_span( )
{
	print_separator( "Using nd_span with C-array from C API" );

	// Get raw C-array from API
	size_t c_array_size;
	double* c_array = get_c_array_from_api( c_array_size );

	// Wrap in nd_span with shape (3, 4)
	nd_span<double> span_from_c( c_array, 3, 4 );

	std::cout << "C-array wrapped as nd_span (3x4):\n";
	for( size_t i = 0; i < 3; ++i )
	{
		for( size_t j = 0; j < 4; ++j )
		{
			std::cout << std::setw( 6 ) << span_from_c( i, j ) << " ";
		}
		std::cout << "\n";
	}

	// Create subspan
	auto sub = span_from_c.subspan( 1, 1, 3 );
	std::cout << "\nSubspan (columns 1-2):\n";
	for( size_t i = 0; i < sub.extent( 0 ); ++i )
	{
		for( size_t j = 0; j < sub.extent( 1 ); ++j )
		{
			std::cout << std::setw( 6 ) << sub( i, j ) << " ";
		}
		std::cout << "\n";
	}

	delete[] c_array;
}

/// \brief Demonstrates wrapping a std::vector with nd_span.
void demo_vector_span( )
{
	print_separator( "Using nd_span with std::vector" );
	std::vector<int> vec_data = { 10, 20, 30, 40, 50, 60 };
	nd_span<int> span_from_vec( vec_data.data( ), 2, 3 );

	std::cout << "Vector wrapped as nd_span (2x3):\n";
	for( size_t i = 0; i < 2; ++i )
	{
		for( size_t j = 0; j < 3; ++j )
		{
			std::cout << std::setw( 4 ) << span_from_vec( i, j ) << " ";
		}
		std::cout << "\n";
	}
}

/// \brief Demonstrates constructing nd_array from a vector of extents.
void demo_array_from_extents( )
{
	print_separator( "Creating nd_array from vector of extents" );
	std::vector<int> extents = { 3, 4, 5 };
	nd_array<int> arr3d_vec( extents );
	std::cout << "Array created with rank: " << arr3d_vec.rank( ) << "\n";
	std::cout << "Array size: " << arr3d_vec.size( ) << "\n";
}

/// \brief Builds and prints a 2D array with sample values.
/// \return The populated 2D array.
nd_array<double> build_and_print_2d_array( )
{
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

	return arr2d;
}

/// \brief Builds and prints a 3D array with sample values.
/// \return The populated 3D array.
nd_array<int> build_and_print_3d_array( )
{
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

	return arr3d;
}

/// \brief Prints basic properties of 2D and 3D arrays.
/// \param arr2d Source 2D array.
/// \param arr3d Source 3D array.
void demo_array_properties( const nd_array<double>& arr2d, const nd_array<int>& arr3d )
{
	print_separator( "Array properties" );
	std::cout << "2D array rank: " << arr2d.rank( ) << "\n";
	std::cout << "2D array size: " << arr2d.size( ) << "\n";
	std::cout << "2D array extent(0): " << arr2d.extent( 0 ) << "\n";
	std::cout << "2D array extent(1): " << arr2d.extent( 1 ) << "\n";
	std::cout << "3D array rank: " << arr3d.rank( ) << "\n";
	std::cout << "3D array size: " << arr3d.size( ) << "\n";
}

/// \brief Demonstrates extracting a single row using subspan.
/// \param arr2d Source 2D array.
void demo_subspan_row( const nd_array<double>& arr2d )
{
	print_separator( "Subspan - getting a row from 2D array" );
	auto row1 = arr2d.subspan( 0, 1, 2 ); // Get row 1 (from index 1 to 2)
	std::cout << "Row 1 of 2D array: ";
	for( size_t j = 0; j < row1.extent( 1 ); ++j )
	{
		std::cout << row1( 0, j ) << " ";
	}
	std::cout << "\n";
}

/// \brief Demonstrates extracting a column range using subspan.
/// \param arr2d Source 2D array.
void demo_subspan_columns( const nd_array<double>& arr2d )
{
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
}

/// \brief Demonstrates slicing a 3D array into a 2D view.
/// \param arr3d Source 3D array.
void demo_slice( const nd_array<int>& arr3d )
{
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
}

/// \brief Demonstrates fill, apply, and copy behaviors.
void demo_fill_apply_copy( )
{
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
}

/// \brief Demonstrates a dynamic-rank array created from an initializer list.
void demo_dynamic_rank_array( )
{
	print_separator( "Dynamic rank array using initializer list" );
	nd_array<float> arr_dynamic( { 2, 3, 2 } );
	std::cout << "Dynamic rank: " << arr_dynamic.rank( ) << "\n";
	std::cout << "Dynamic size: " << arr_dynamic.size( ) << "\n";
}

/// \brief Demonstrates reshape, transpose, flatten, and squeeze.
void demo_shape_operations( )
{
	print_separator( "Shape operations" );
	nd_array<int> arr( 2, 3 );
	int value = 0;
	for( auto& v: arr )
	{
		v = value++;
	}

	auto reshaped = arr.reshape( 3, 2 );
	std::cout << "Reshape (3x2) element [1,0]: " << reshaped( 1, 0 ) << "\n";

	auto flat = arr.flatten( );
	std::cout << "Flatten size: " << flat.extent( 0 ) << "\n";

	auto transposed = arr.transpose( { 1, 0 } );
	std::cout << "Transpose [1,0] from [0,1]: " << transposed( 1, 0 ) << "\n";

	nd_array<int> squeezed_source( { 1, 3, 1, 2 } );
	auto squeezed = squeezed_source.squeeze( );
	std::cout << "Squeeze rank: " << squeezed.rank( ) << "\n";
}

/// \brief Demonstrates deep copy from nd_span.
void demo_copy_from_span( )
{
	print_separator( "Deep copy from nd_span" );
	nd_array<int> arr( 2, 3 );
	arr.fill( 7 );
	nd_span<int> span( arr.data( ), 2, 3 );

	nd_array<int> copy = nd_array<int>::from_span( span );
	arr.fill( 9 );
	std::cout << "Copy[0,0] after original change: " << copy( 0, 0 ) << "\n";
}

void demo_iterator_access( )
{
	print_separator( "Iterator access" );
	nd_array<int> arr( 2, 2 );
	arr.fill( 5 );

	std::cout << "Iterating with non-const iterator:\n";
	for( auto& v: arr )
	{
		std::cout << v << " ";
	}
	std::cout << "\n";

	std::cout << "Modifying elements through iterator:\n";
	for( auto& v: arr )
	{
		v += 1;
		std::cout << v << " ";
	}

	const nd_array<int>& carr = arr;
	std::cout << "Iterating with const iterator:\n";
	for( auto& v: carr )
	{
		std::cout << v << " ";
	}
	std::cout << "\n";
}

int main( )
{
	demo_c_api_span( );
	demo_vector_span( );
	demo_array_from_extents( );

	nd_array<double> arr2d = build_and_print_2d_array( );
	nd_array<int> arr3d    = build_and_print_3d_array( );

	demo_array_properties( arr2d, arr3d );
	demo_subspan_row( arr2d );
	demo_subspan_columns( arr2d );
	demo_slice( arr3d );
	demo_fill_apply_copy( );
	demo_dynamic_rank_array( );
	demo_shape_operations( );
	demo_copy_from_span( );
	demo_iterator_access( );

	return 0;
}
