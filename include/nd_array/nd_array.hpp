#pragma once

#include <algorithm>
#include <array>
#include <initializer_list>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <type_traits>
#include <vector>

/// \namespace cppa
/// \brief C++ Array namespace containing n-dimensional array and span classes
namespace cppa
{

	/// \namespace cppa::detail
	/// \brief Internal implementation details, not for public use
	namespace detail
	{
		/// \brief Helper struct for computing linear offsets from multi-dimensional indices
		/// \tparam MaxRank Maximum number of dimensions supported
		template<size_t MaxRank>
		struct offset_computer
		{
			using size_type = size_t;

			/// \brief Computes the linear offset from multi-dimensional indices
			/// \tparam Indices Variadic index types (typically size_t)
			/// \param extents Array of dimension sizes
			/// \param strides Array of stride values for each dimension
			/// \param indices Variable number of indices, one per dimension
			/// \return Linear offset into contiguous memory
			/// \throws std::out_of_range if any index is out of bounds
			template<typename... Indices>
			[[nodiscard]] static constexpr size_type compute( const std::array<size_type, MaxRank>& extents, const std::array<size_type, MaxRank>& strides,
			                                                  Indices... indices )
			{
				size_type idx[]  = { static_cast<size_type>( indices )... };
				size_type offset = 0;
				for( size_t i = 0; i < sizeof...( indices ); ++i )
				{
					if( idx[i] >= extents[i] )
					{
						throw std::out_of_range( "Index out of bounds" );
					}
					offset += idx[i] * strides[i];
				}
				return offset;
			}
		};

		/// \brief Helper struct for computing stride values from extents
		/// \tparam MaxRank Maximum number of dimensions supported
		template<size_t MaxRank>
		struct stride_computer
		{
			using size_type = size_t;

			/// \brief Computes row-major strides from dimension extents
			/// \param strides Output array to store computed strides
			/// \param extents Array of dimension sizes
			/// \param rank Actual number of dimensions in use
			/// \note Uses row-major (C-style) ordering where last dimension varies fastest
			static constexpr void compute( std::array<size_type, MaxRank>& strides, const std::array<size_type, MaxRank>& extents, size_type rank ) noexcept
			{
				if( rank == 0 )
					return;

				strides[rank - 1] = 1;
				for( size_t i = rank - 1; i > 0; --i )
				{
					strides[i - 1] = strides[i] * extents[i];
				}

				for( size_t i = rank; i < MaxRank; ++i )
				{
					strides[i] = 0;
				}
			}
		};

		template<typename SizeType>
		struct extents_view
		{
			const SizeType* data = nullptr;
			size_t size          = 0;

			const SizeType* begin( ) const noexcept { return data; }
			const SizeType* end( ) const noexcept { return data + size; }
		};

		/// \brief Computes total number of elements from extents
		/// \tparam MaxRank Maximum number of dimensions supported
		/// \param extents Extents array
		/// \param rank Number of active dimensions
		/// \return Product of active extents (0 when rank is 0)
		template<size_t MaxRank>
		[[nodiscard]] constexpr size_t compute_size( const std::array<size_t, MaxRank>& extents, size_t rank ) noexcept
		{
			if( rank == 0 )
				return 0;
			size_t s = 1;
			for( size_t i = 0; i < rank; ++i )
			{
				s *= extents[i];
			}
			return s;
		}

		/// \brief Checks if a span is contiguous in row-major order
		/// \tparam MaxRank Maximum number of dimensions supported
		/// \param extents Extents array
		/// \param strides Strides array
		/// \param rank Number of active dimensions
		/// \return True if the view is contiguous
		template<size_t MaxRank>
		[[nodiscard]] constexpr bool is_contiguous( const std::array<size_t, MaxRank>& extents, const std::array<size_t, MaxRank>& strides, size_t rank ) noexcept
		{
			if( rank == 0 )
				return true;
			if( compute_size<MaxRank>( extents, rank ) == 0 )
				return true;
			if( strides[rank - 1] != 1 )
				return false;
			for( size_t i = rank - 1; i > 0; --i )
			{
				if( strides[i - 1] != strides[i] * extents[i] )
				{
					return false;
				}
			}
			return true;
		}

		/// \brief Validates a permutation for transpose
		/// 	param MaxRank Maximum number of dimensions supported
		/// \param axes Permutation array
		/// \param rank Number of active dimensions
		/// 	hrows std::invalid_argument if the permutation is invalid
		template<size_t MaxRank>
		inline void validate_permutation( const size_t* axes, size_t rank )
		{
			if( rank > MaxRank )
			{
				throw std::invalid_argument( "Permutation size must be <= MaxRank" );
			}
			std::array<bool, MaxRank> seen{};
			for( size_t i = 0; i < rank; ++i )
			{
				const size_t axis = axes[i];
				if( axis >= rank || seen[axis] )
				{
					throw std::invalid_argument( "Invalid permutation" );
				}
				seen[axis] = true;
			}
		}
	} // namespace detail

	/// \class nd_span
	/// \brief Non-owning view over multi-dimensional data with dynamic rank
	/// \tparam T Element type
	/// \tparam MaxRank Maximum number of dimensions (default: 8)
	///
	/// nd_span provides a lightweight, non-owning reference to multi-dimensional data.
	/// It's similar to std::span but for multiple dimensions, like C++23's std::mdspan.
	/// The actual rank is determined at runtime but cannot exceed MaxRank.
	///
	/// <b>Memory Layout</b>
	///
	/// Data is assumed to be in row-major (C-style) order where the last dimension
	/// varies fastest in memory.
	///
	/// <b>Typical Usage</b>
	///
	/// \code
	/// double data[12];
	/// nd_span<double> span(data, 3, 4);  // 3x4 matrix
	/// span(1, 2) = 5.0;  // Access element at row 1, column 2
	/// \endcode
	template<typename Ty, size_t MaxRank = 8>
	class nd_span
	{
	public:
		using value_type      = Ty;        ///< Type of elements
		using size_type       = size_t;   ///< Type for sizes and indices
		using reference       = Ty&;       ///< Reference to element
		using const_reference = const Ty&; ///< Const reference to element
		using pointer         = Ty*;       ///< Pointer to element
		using const_pointer   = const Ty*; ///< Const pointer to element

		/// \brief Constructs a span from raw data with explicit extents and strides
		/// \param data Pointer to the first element
		/// \param extents Sizes of each dimension
		/// \param strides Stride values for each dimension
		/// \param rank Number of dimensions (must be <= MaxRank)
		/// \note This is the primary constructor used by subspan/slice operations
		constexpr nd_span( pointer data, const std::array<size_type, MaxRank>& extents, const std::array<size_type, MaxRank>& strides, size_type rank ) noexcept
		    : data_( data )
		    , extents_( extents )
		    , strides_( strides )
		    , rank_( rank )
		{
		}

		/// \brief Constructs a span from raw data with dimension sizes
		/// \param data Pointer to the first element
		/// \param extents Initializer list of dimension sizes {dim0, dim1, ...}
		/// \throws std::invalid_argument if number of dimensions exceeds MaxRank
		/// \example
		/// \code
		/// double data[12];
		/// nd_span<double> span(data, {3, 4});  // 3x4 matrix
		/// \endcode
		nd_span( pointer data, std::initializer_list<size_type> extents ) : data_( data ), rank_( extents.size( ) )
		{
			if( rank_ > MaxRank )
			{
				throw std::invalid_argument( "Rank exceeds MaxRank" );
			}

			size_t idx = 0;
			for( auto extent: extents )
			{
				extents_[idx++] = extent;
			}
			for( size_t i = rank_; i < MaxRank; ++i )
			{
				extents_[i] = 0;
			}

			compute_strides( );
		}

		/// \brief Constructs a span from raw data with a container of dimension sizes
		/// \tparam Container Type of container holding extents (e.g., std::vector<size_t>)
		/// \param data Pointer to the first element
		/// \param extents Container with dimension sizes
		/// \throws std::invalid_argument if number of dimensions exceeds MaxRank
		/// \example
		/// \code
		/// double data[24];
		/// std::vector<size_t> dims = {2, 3, 4};
		/// nd_span<double> span(data, dims);  // 2x3x4 array
		/// \endcode
		template<typename Container>
		nd_span( pointer data, const Container& extents,
		         typename std::enable_if<!std::is_integral<Container>::value && !std::is_same<Container, nd_span>::value>::type* = nullptr )
		    : data_( data )
		    , rank_( extents.size( ) )
		{
			if( rank_ > MaxRank )
			{
				throw std::invalid_argument( "Rank exceeds MaxRank" );
			}

			size_t idx = 0;
			for( auto extent: extents )
			{
				extents_[idx++] = static_cast<size_type>( extent );
			}
			for( size_t i = rank_; i < MaxRank; ++i )
			{
				extents_[i] = 0;
			}

			compute_strides( );
		}

		/// \brief Constructs a span from raw data with variadic dimension sizes
		/// \tparam Indices Variadic index types (typically size_t or convertible to size_t)
		/// \param data Pointer to the first element
		/// \param indices Dimension sizes as separate arguments
		/// \example
		/// \code
		/// double data[24];
		/// nd_span<double> span(data, 2, 3, 4);  // 2x3x4 array
		/// \endcode
		template<typename... Indices>
		nd_span( pointer data, Indices... indices ) : data_( data )
		                                            , rank_( sizeof...( indices ) )
		{
			static_assert( sizeof...( indices ) <= MaxRank, "Too many dimensions" );

			size_type temp[] = { static_cast<size_type>( indices )... };
			for( size_t i = 0; i < rank_; ++i )
			{
				extents_[i] = temp[i];
			}
			for( size_t i = rank_; i < MaxRank; ++i )
			{
				extents_[i] = 0;
			}

			compute_strides( );
		}

		/// \brief Accesses an element with multi-dimensional indexing (non-const)
		/// \tparam Indices Variadic index types (typically size_t)
		/// \param indices Multi-dimensional indices (i, j, k, ...)
		/// \return Reference to the element at the specified location
		/// \throws std::out_of_range if any index is out of bounds
		/// \example
		/// \code
		/// nd_span<double> span(data, 3, 4);
		/// span(1, 2) = 5.0;  // Set element at row 1, column 2
		/// \endcode
		template<typename... Indices>
		[[nodiscard]] reference operator( )( Indices... indices )
		{
			return data_[detail::offset_computer<MaxRank>::compute( extents_, strides_, indices... )];
		}

		/// \brief Accesses an element with multi-dimensional indexing (const)
		/// \tparam Indices Variadic index types (typically size_t)
		/// \param indices Multi-dimensional indices (i, j, k, ...)
		/// \return Const reference to the element at the specified location
		/// \throws std::out_of_range if any index is out of bounds
		template<typename... Indices>
		[[nodiscard]] const_reference operator( )( Indices... indices ) const
		{
			return data_[detail::offset_computer<MaxRank>::compute( extents_, strides_, indices... )];
		}

		/// \brief Creates a subspan by restricting a range along one dimension
		/// \param dim Dimension to restrict (0-based)
		/// \param start Starting index in that dimension (inclusive)
		/// \param end Ending index in that dimension (exclusive)
		/// \return New nd_span view of the restricted data
		/// \throws std::out_of_range if dimension or range is invalid
		/// \example
		/// \code
		/// nd_span<double> span(data, 5, 10);
		/// auto sub = span.subspan(0, 1, 4);  // Rows 1-3, all columns
		/// \endcode
		[[nodiscard]] nd_span subspan( size_type dim, size_type start, size_type end ) const
		{
			if( dim >= rank_ )
			{
				throw std::out_of_range( "Dimension out of range" );
			}
			if( start >= extents_[dim] || end > extents_[dim] || start >= end )
			{
				throw std::out_of_range( "Invalid range for subspan" );
			}

			std::array<size_type, MaxRank> new_extents = extents_;
			new_extents[dim]                           = end - start;

			size_type offset = start * strides_[dim];

			return nd_span( data_ + offset, new_extents, strides_, rank_ );
		}

		/// \brief Creates a lower-dimensional view by fixing one dimension's index
		/// \param dim Dimension to slice (0-based)
		/// \param index Index value to fix for that dimension
		/// \return New nd_span with rank reduced by 1
		/// \throws std::out_of_range if dimension or index is invalid
		/// \example
		/// \code
		/// nd_span<double> span(data, 3, 4, 5);  // 3D array
		/// auto slice = span.slice(0, 1);         // 2D array (4x5) at first dimension index 1
		/// \endcode
		[[nodiscard]] nd_span slice( size_type dim, size_type index ) const
		{
			if( dim >= rank_ )
			{
				throw std::out_of_range( "Dimension out of range" );
			}
			if( index >= extents_[dim] )
			{
				throw std::out_of_range( "Index out of bounds" );
			}

			std::array<size_type, MaxRank> new_extents;
			std::array<size_type, MaxRank> new_strides;

			size_type new_rank = rank_ - 1;
			size_type offset   = index * strides_[dim];

			size_type j = 0;
			for( size_t i = 0; i < rank_; ++i )
			{
				if( i != dim )
				{
					new_extents[j] = extents_[i];
					new_strides[j] = strides_[i];
					++j;
				}
			}

			for( size_t i = new_rank; i < MaxRank; ++i )
			{
				new_extents[i] = 0;
				new_strides[i] = 0;
			}

			return nd_span( data_ + offset, new_extents, new_strides, new_rank );
		}

		/// \brief Reshapes the span (view-only, row-major contiguous required)
		/// \param new_extents New shape extents
		/// \return Reshaped view
		/// \throws std::invalid_argument if rank exceeds MaxRank or size mismatch
		/// \throws std::runtime_error if the view is not contiguous
		[[nodiscard]] nd_span reshape( std::initializer_list<size_type> new_extents ) const
		{
			return reshape_impl( new_extents.begin( ), new_extents.size( ) );
		}

		/// \brief Reshapes the span with variadic extents (view-only, row-major contiguous required)
		/// \tparam Indices Variadic extent types
		/// \param new_extents New shape extents
		/// \return Reshaped view
		template<typename... Indices>
		[[nodiscard]] nd_span reshape( Indices... new_extents ) const
		{
			size_type temp[] = { static_cast<size_type>( new_extents )... };
			return reshape_impl( temp, sizeof...( new_extents ) );
		}

		/// \brief Returns a transposed view using an axis permutation
		/// \param axes Permutation of axes
		/// \return Transposed view
		/// 	hrows std::invalid_argument if permutation is invalid
		[[nodiscard]] nd_span transpose( std::initializer_list<size_type> axes ) const
		{
			return transpose_impl( axes.begin( ), axes.size( ) );
		}

		/// \brief Returns a transposed view by swapping the last two axes
		/// \return Transposed view
		[[nodiscard]] nd_span T( ) const
		{
			size_type axes_rank = 0;
			auto axes = make_T_axes( axes_rank );
			return transpose_impl( axes.data( ), axes_rank );
		}

		/// \brief Flattens the span into a 1D view (row-major contiguous required)
		/// \return 1D view of the data
		[[nodiscard]] nd_span flatten( ) const { return reshape( size( ) ); }

		/// \brief Removes dimensions of extent 1
		/// \return View with singleton dimensions removed
		[[nodiscard]] nd_span squeeze( ) const
		{
			std::array<size_type, MaxRank> new_extents;
			std::array<size_type, MaxRank> new_strides;
			size_type new_rank = 0;

			for( size_t i = 0; i < rank_; ++i )
			{
				if( extents_[i] != 1 )
				{
					new_extents[new_rank] = extents_[i];
					new_strides[new_rank] = strides_[i];
					++new_rank;
				}
			}

			for( size_t i = new_rank; i < MaxRank; ++i )
			{
				new_extents[i] = 0;
				new_strides[i] = 0;
			}

			return nd_span( data_, new_extents, new_strides, new_rank );
		}

		/// \brief Gets the size of a specific dimension
		/// \param dim Dimension index (0-based)
		/// \return Size of the specified dimension
		/// \throws std::out_of_range if dimension is >= rank
		[[nodiscard]] size_type extent( size_type dim ) const
		{
			if( dim >= rank_ )
			{
				throw std::out_of_range( "Dimension out of range" );
			}
			return extents_[dim];
		}

		/// \brief Gets the stride of a specific dimension
		/// \param dim Dimension index (0-based)
		/// \return Stride of the specified dimension
		/// \throws std::out_of_range if dimension is >= rank
		[[nodiscard]] size_type stride( size_type dim ) const
		{
			if( dim >= rank_ )
			{
				throw std::out_of_range( "Dimension out of range" );
			}
			return strides_[dim];
		}

		/// \brief Gets the active extents as a view sized to rank()
		/// \return View of extents for the active dimensions
		[[nodiscard]] detail::extents_view<size_type> extents( ) const noexcept { return { extents_.data( ), rank_ }; }

		/// \brief Gets the total number of elements in the span
		/// \return Total number of elements (product of all extents)
		[[nodiscard]] size_type size( ) const noexcept { return compute_size( ); }

		/// \brief Gets the number of dimensions
		/// \return Current rank (number of dimensions)
		[[nodiscard]] size_type rank( ) const noexcept { return rank_; }

		/// \brief Gets the maximum number of dimensions supported
		/// \return MaxRank template parameter
		[[nodiscard]] static constexpr size_type max_rank( ) noexcept { return MaxRank; }

		/// \brief Gets a pointer to the underlying data (non-const)
		/// \return Pointer to the first element
		[[nodiscard]] pointer data( ) noexcept { return data_; }

		/// \brief Gets a pointer to the underlying data (const)
		/// \return Const pointer to the first element
		[[nodiscard]] const_pointer data( ) const noexcept { return data_; }

		/// \brief Returns a pointer to the first element for flat iteration
		[[nodiscard]] pointer begin( ) noexcept { return data_; }

		/// \brief Returns a pointer past the last element for flat iteration
		[[nodiscard]] pointer end( ) noexcept { return data_ + size( ); }

		/// \brief Returns a const pointer to the first element for flat iteration
		[[nodiscard]] const_pointer begin( ) const noexcept { return data_; }

		/// \brief Returns a const pointer past the last element for flat iteration
		[[nodiscard]] const_pointer end( ) const noexcept { return data_ + size( ); }

		/// \brief Returns a const pointer to the first element for flat iteration
		[[nodiscard]] const_pointer cbegin( ) const noexcept { return data_; }

		/// \brief Returns a const pointer past the last element for flat iteration
		[[nodiscard]] const_pointer cend( ) const noexcept { return data_ + size( ); }

	private:
		pointer data_;                           ///< Pointer to the first element
		std::array<size_type, MaxRank> extents_; ///< Size of each dimension
		std::array<size_type, MaxRank> strides_; ///< Stride for each dimension
		size_type rank_;                         ///< Actual number of dimensions

		/// \brief Computes row-major strides from extents
		constexpr void compute_strides( ) noexcept { detail::stride_computer<MaxRank>::compute( strides_, extents_, rank_ ); }

		/// \brief Computes total number of elements from extents
		[[nodiscard]] constexpr size_type compute_size( ) const noexcept
		{
			if( rank_ == 0 )
				return 0;
			size_type s = 1;
			for( size_t i = 0; i < rank_; ++i )
			{
				s *= extents_[i];
			}
			return s;
		}

		[[nodiscard]] nd_span reshape_impl( const size_type* new_extents, size_type new_rank ) const
		{
			if( new_rank > MaxRank )
			{
				throw std::invalid_argument( "Rank exceeds MaxRank" );
			}
			if( !detail::is_contiguous<MaxRank>( extents_, strides_, rank_ ) )
			{
				throw std::runtime_error( "Reshape requires contiguous data" );
			}

			std::array<size_type, MaxRank> new_extents_array{};
			for( size_t i = 0; i < new_rank; ++i )
			{
				new_extents_array[i] = new_extents[i];
			}

			const size_type new_size = detail::compute_size<MaxRank>( new_extents_array, new_rank );
			if( new_size != size( ) )
			{
				throw std::invalid_argument( "Reshape size mismatch" );
			}

			std::array<size_type, MaxRank> new_strides{};
			detail::stride_computer<MaxRank>::compute( new_strides, new_extents_array, new_rank );

			return nd_span( data_, new_extents_array, new_strides, new_rank );
		}

		[[nodiscard]] nd_span transpose_impl( const size_type* axes, size_type axes_rank ) const
		{
			if( axes_rank != rank_ )
			{
				throw std::invalid_argument( "Permutation size must match rank" );
			}
			detail::validate_permutation<MaxRank>( axes, axes_rank );

			std::array<size_type, MaxRank> new_extents;
			std::array<size_type, MaxRank> new_strides;
			for( size_t i = 0; i < rank_; ++i )
			{
				new_extents[i] = extents_[axes[i]];
				new_strides[i] = strides_[axes[i]];
			}
			for( size_t i = rank_; i < MaxRank; ++i )
			{
				new_extents[i] = 0;
				new_strides[i] = 0;
			}

			return nd_span( data_, new_extents, new_strides, rank_ );
		}

		[[nodiscard]] std::array<size_type, MaxRank> make_T_axes( size_type& axes_rank ) const
		{
			axes_rank = rank_;
			std::array<size_type, MaxRank> axes{};
			for( size_t i = 0; i < rank_; ++i )
			{
				axes[i] = i;
			}
			if( rank_ >= 2 )
			{
				std::swap( axes[rank_ - 1], axes[rank_ - 2] );
			}
			return axes;
		}
	};

	/// \class nd_array
	/// \brief Owning multi-dimensional array with dynamic rank and single memory allocation
	/// \tparam T Element type
	/// \tparam MaxRank Maximum number of dimensions (default: 8)
	///
	/// nd_array provides a dynamically-sized multi-dimensional array with:
	///
	/// - Single memory allocation (on construction only)
	/// - Runtime-determined rank (up to MaxRank)
	/// - Runtime-determined extents for each dimension
	/// - Row-major (C-style) memory layout
	/// - mdspan-like interface similar to C++23
	///
	/// <b>Memory Allocation</b>
	///
	/// All memory is allocated once during construction using `std::unique_ptr<T[]>`.
	/// No further allocations occur during the object's lifetime, minimizing
	/// allocation overhead and memory fragmentation.
	///
	/// <b>Typical Usage</b>
	///
	/// \code
	/// nd_array<double> matrix(3, 4);          // 3x4 matrix
	/// matrix.fill(0.0);                       // Fill with zeros
	/// matrix(1, 2) = 5.0;                     // Set element
	/// auto sub = matrix.subspan(0, 1, 3);     // View of rows 1-2
	/// \endcode
	template<typename Ty, size_t MaxRank = 8>
	class nd_array
	{
	public:
		using value_type      = Ty;        ///< Type of elements
		using size_type       = size_t;   ///< Type for sizes and indices
		using reference       = Ty&;       ///< Reference to element
		using const_reference = const Ty&; ///< Const reference to element
		using pointer         = Ty*;       ///< Pointer to element
		using const_pointer   = const Ty*; ///< Const pointer to element

		/// \brief Constructs an empty array with no dimensions
		/// \note No memory is allocated
		nd_array( ) noexcept : data_( nullptr ), size_( 0 ), rank_( 0 )
		{
			extents_.fill( 0 );
			strides_.fill( 0 );
		}

		/// \brief Constructs an array with specified dimension sizes
		/// \param extents Initializer list of dimension sizes {dim0, dim1, ...}
		/// \throws std::invalid_argument if number of dimensions exceeds MaxRank
		///
		/// <b>Example</b>
		///
		/// \code
		/// nd_array<double> arr({3, 4, 5});  // 3x4x5 array
		/// \endcode
		nd_array( std::initializer_list<size_type> extents ) : rank_( extents.size( ) )
		{
			if( rank_ > MaxRank )
			{
				throw std::invalid_argument( "Rank exceeds MaxRank" );
			}

			size_t idx = 0;
			for( auto extent: extents )
			{
				extents_[idx++] = extent;
			}
			for( size_t i = rank_; i < MaxRank; ++i )
			{
				extents_[i] = 0;
			}

			compute_strides( );
			size_ = compute_size( );
			data_ = std::make_unique<Ty[]>( size_ );
		}

		/// \brief Constructs an array from a container of dimension sizes
		/// \tparam Container Type of container holding extents (e.g., std::vector<size_t>)
		/// \param extents Container with dimension sizes
		/// \throws std::invalid_argument if number of dimensions exceeds MaxRank
		/// \example
		/// \code
		/// std::vector<size_t> dims = {2, 3, 4};
		/// nd_array<double> arr(dims);  // 2x3x4 array
		/// \endcode
		template<typename Container>
		nd_array( const Container& extents, typename std::enable_if<!std::is_integral<Container>::value && !std::is_same<Container, nd_array>::value>::type* = nullptr )
		    : rank_( extents.size( ) )
		{
			if( rank_ > MaxRank )
			{
				throw std::invalid_argument( "Rank exceeds MaxRank" );
			}

			size_t idx = 0;
			for( auto extent: extents )
			{
				extents_[idx++] = static_cast<size_type>( extent );
			}
			for( size_t i = rank_; i < MaxRank; ++i )
			{
				extents_[i] = 0;
			}

			compute_strides( );
			size_ = compute_size( );
			data_ = std::make_unique<Ty[]>( size_ );
		}

		/// \brief Constructs an array with variadic dimension sizes
		/// \tparam Indices Variadic index types (typically size_t or convertible to size_t)
		/// \param indices Dimension sizes as separate arguments
		/// \example
		/// \code
		/// nd_array<double> arr(2, 3, 4);  // 2x3x4 array
		/// \endcode
		template<typename... Indices>
		nd_array( Indices... indices ) : rank_( sizeof...( indices ) )
		{
			static_assert( sizeof...( indices ) <= MaxRank, "Too many dimensions" );

			size_type temp[] = { static_cast<size_type>( indices )... };
			for( size_t i = 0; i < rank_; ++i )
			{
				extents_[i] = temp[i];
			}
			for( size_t i = rank_; i < MaxRank; ++i )
			{
				extents_[i] = 0;
			}

			compute_strides( );
			size_ = compute_size( );
			data_ = std::make_unique<Ty[]>( size_ );
		}

		/// \brief Copy constructor - performs deep copy of data
		/// \param other Array to copy from
		nd_array( const nd_array& other ) : extents_( other.extents_ ), strides_( other.strides_ ), size_( other.size_ ), rank_( other.rank_ )
		{
			if( size_ > 0 )
			{
				data_ = std::make_unique<Ty[]>( size_ );
				std::copy( other.data_.get( ), other.data_.get( ) + size_, data_.get( ) );
			}
		}

		/// \brief Move constructor - transfers ownership of data
		nd_array( nd_array&& other ) noexcept = default;

		/// \brief Constructs an owning array by deep-copying an nd_span
		/// \param span Source span to copy
		/// 	hrows std::invalid_argument if span rank exceeds MaxRank
		explicit nd_array( const nd_span<const Ty, MaxRank>& span ) : nd_array( from_span( span ) ) {}

		/// \brief Constructs an owning array by deep-copying an nd_span
		/// \param span Source span to copy
		explicit nd_array( const nd_span<Ty, MaxRank>& span ) : nd_array( from_span( span ) ) {}

		/// \brief Copy assignment operator - performs deep copy of data
		/// \param other Array to copy from
		/// \return Reference to this array
		nd_array& operator=( const nd_array& other )
		{
			if( this != &other )
			{
				rank_    = other.rank_;
				size_    = other.size_;
				extents_ = other.extents_;
				strides_ = other.strides_;
				if( size_ > 0 )
				{
					data_ = std::make_unique<Ty[]>( size_ );
					std::copy( other.data_.get( ), other.data_.get( ) + size_, data_.get( ) );
				}
				else
				{
					data_.reset( );
				}
			}
			return *this;
		}

		/// \brief Move assignment operator - transfers ownership of data
		/// \param other Array to move from
		/// \return Reference to this array
		nd_array& operator=( nd_array&& other ) noexcept = default;

		/// \brief Assigns from an nd_span by deep-copying its contents
		/// \param span Source span to copy
		/// \return Reference to this array
		nd_array& operator=( const nd_span<const Ty, MaxRank>& span ) { return *this = from_span( span ); }

		/// \brief Assigns from an nd_span by deep-copying its contents
		/// \param span Source span to copy
		/// \return Reference to this array
		nd_array& operator=( const nd_span<Ty, MaxRank>& span ) { return *this = from_span( span ); }

		/// \brief Creates an owning array by deep-copying an nd_span
		/// \param span Source span to copy
		/// \return Newly allocated array with the same contents
		/// 	hrows std::invalid_argument if span rank exceeds MaxRank
		static nd_array from_span( const nd_span<const Ty, MaxRank>& span ) { return from_span_impl( span ); }

		/// \brief Creates an owning array by deep-copying an nd_span
		/// \param span Source span to copy
		/// \return Newly allocated array with the same contents
		static nd_array from_span( const nd_span<Ty, MaxRank>& span ) { return from_span_impl( span ); }

		/// \brief Accesses an element with multi-dimensional indexing (non-const)
		/// \tparam Indices Variadic index types (typically size_t)
		/// \param indices Multi-dimensional indices (i, j, k, ...)
		/// \return Reference to the element at the specified location
		/// \throws std::out_of_range if any index is out of bounds
		/// \example
		/// \code
		/// nd_array<double> arr(3, 4);
		/// arr(1, 2) = 5.0;  // Set element at row 1, column 2
		/// \endcode
		template<typename... Indices>
		[[nodiscard]] reference operator( )( Indices... indices )
		{
			static_assert( sizeof...( indices ) <= MaxRank, "Too many indices" );
			return data_[detail::offset_computer<MaxRank>::compute( extents_, strides_, indices... )];
		}

		/// \brief Accesses an element with multi-dimensional indexing (const)
		/// \tparam Indices Variadic index types (typically size_t)
		/// \param indices Multi-dimensional indices (i, j, k, ...)
		/// \return Const reference to the element at the specified location
		/// \throws std::out_of_range if any index is out of bounds
		template<typename... Indices>
		[[nodiscard]] const_reference operator( )( Indices... indices ) const
		{
			static_assert( sizeof...( indices ) <= MaxRank, "Too many indices" );
			return data_[detail::offset_computer<MaxRank>::compute( extents_, strides_, indices... )];
		}

		/// \brief Creates a subspan with multiple dimension ranges
		/// \param ranges Initializer list of {start, end} pairs for each dimension
		/// \return Non-owning view (nd_span) of the restricted data
		/// \throws std::out_of_range if too many dimensions or invalid ranges
		/// \example
		/// \code
		/// nd_array<double> arr(5, 10);
		/// auto sub = arr.subspan({{1, 4}, {2, 8}});  // Rows 1-3, columns 2-7
		/// \endcode
		[[nodiscard]] nd_span<Ty, MaxRank> subspan( std::initializer_list<std::pair<size_type, size_type>> ranges )
		{
			std::array<size_type, MaxRank> new_extents = extents_;
			std::array<size_type, MaxRank> new_strides = strides_;
			size_type offset                           = 0;
			size_type dim                              = 0;

			for( const auto& [start, end]: ranges )
			{
				if( dim >= rank_ )
				{
					throw std::out_of_range( "Too many dimensions in subspan" );
				}
				if( start >= extents_[dim] || end > extents_[dim] || start >= end )
				{
					throw std::out_of_range( "Invalid range for subspan" );
				}
				offset += start * strides_[dim];
				new_extents[dim] = end - start;
				++dim;
			}

			return nd_span<Ty, MaxRank>( data_.get( ) + offset, new_extents, new_strides, rank_ );
		}

		/// \brief Creates a subspan with multiple dimension ranges (const)
		/// \param ranges Initializer list of {start, end} pairs for each dimension
		/// \return Non-owning view (nd_span) of the restricted data
		/// 	hrows std::out_of_range if too many dimensions or invalid ranges
		[[nodiscard]] nd_span<const Ty, MaxRank> subspan( std::initializer_list<std::pair<size_type, size_type>> ranges ) const
		{
			std::array<size_type, MaxRank> new_extents = extents_;
			std::array<size_type, MaxRank> new_strides = strides_;
			size_type offset                           = 0;
			size_type dim                              = 0;

			for( const auto& [start, end]: ranges )
			{
				if( dim >= rank_ )
				{
					throw std::out_of_range( "Too many dimensions in subspan" );
				}
				if( start >= extents_[dim] || end > extents_[dim] || start >= end )
				{
					throw std::out_of_range( "Invalid range for subspan" );
				}
				offset += start * strides_[dim];
				new_extents[dim] = end - start;
				++dim;
			}

			return nd_span<const Ty, MaxRank>( data_.get( ) + offset, new_extents, new_strides, rank_ );
		}

		/// \brief Creates a subspan by restricting a range along one dimension
		/// \param dim Dimension to restrict (0-based)
		/// \param start Starting index in that dimension (inclusive)
		/// \param end Ending index in that dimension (exclusive)
		/// \return Non-owning view (nd_span) of the restricted data
		/// \throws std::out_of_range if dimension or range is invalid
		/// \example
		/// \code
		/// nd_array<double> arr(5, 10);
		/// auto sub = arr.subspan(0, 1, 4);  // Rows 1-3, all columns
		/// \endcode
		[[nodiscard]] nd_span<Ty, MaxRank> subspan( size_type dim, size_type start, size_type end )
		{
			if( dim >= rank_ )
			{
				throw std::out_of_range( "Dimension out of range" );
			}
			if( start >= extents_[dim] || end > extents_[dim] || start >= end )
			{
				throw std::out_of_range( "Invalid range for subspan" );
			}

			std::array<size_type, MaxRank> new_extents = extents_;
			new_extents[dim]                           = end - start;

			size_type offset = start * strides_[dim];

			return nd_span<Ty, MaxRank>( data_.get( ) + offset, new_extents, strides_, rank_ );
		}

		/// \brief Creates a subspan by restricting a range along one dimension (const)
		/// \param dim Dimension to restrict (0-based)
		/// \param start Starting index in that dimension (inclusive)
		/// \param end Ending index in that dimension (exclusive)
		/// \return Non-owning view (nd_span) of the restricted data
		/// 	hrows std::out_of_range if dimension or range is invalid
		[[nodiscard]] nd_span<const Ty, MaxRank> subspan( size_type dim, size_type start, size_type end ) const
		{
			if( dim >= rank_ )
			{
				throw std::out_of_range( "Dimension out of range" );
			}
			if( start >= extents_[dim] || end > extents_[dim] || start >= end )
			{
				throw std::out_of_range( "Invalid range for subspan" );
			}

			std::array<size_type, MaxRank> new_extents = extents_;
			new_extents[dim]                           = end - start;

			size_type offset = start * strides_[dim];

			return nd_span<const T, MaxRank>( data_.get( ) + offset, new_extents, strides_, rank_ );
		}

		/// \brief Creates a lower-dimensional view by fixing one dimension's index
		/// \param dim Dimension to slice (0-based)
		/// \param index Index value to fix for that dimension
		/// \return Non-owning view (nd_span) with rank reduced by 1
		/// \throws std::out_of_range if dimension or index is invalid
		/// \example
		/// \code
		/// nd_array<double> arr(3, 4, 5);  // 3D array
		/// auto slice = arr.slice(0, 1);    // 2D array (4x5) at first dimension index 1
		/// \endcode
		[[nodiscard]] nd_span<Ty, MaxRank> slice( size_type dim, size_type index )
		{
			if( dim >= rank_ )
			{
				throw std::out_of_range( "Dimension out of range" );
			}
			if( index >= extents_[dim] )
			{
				throw std::out_of_range( "Index out of bounds" );
			}

			std::array<size_type, MaxRank> new_extents;
			std::array<size_type, MaxRank> new_strides;

			size_type new_rank = rank_ - 1;
			size_type offset   = index * strides_[dim];

			size_type j = 0;
			for( size_t i = 0; i < rank_; ++i )
			{
				if( i != dim )
				{
					new_extents[j] = extents_[i];
					new_strides[j] = strides_[i];
					++j;
				}
			}

			for( size_t i = new_rank; i < MaxRank; ++i )
			{
				new_extents[i] = 0;
				new_strides[i] = 0;
			}

			return nd_span<Ty, MaxRank>( data_.get( ) + offset, new_extents, new_strides, new_rank );
		}

		/// \brief Creates a lower-dimensional view by fixing one dimension's index (const)
		/// \param dim Dimension to slice (0-based)
		/// \param index Index value to fix for that dimension
		/// \return Non-owning view (nd_span) with rank reduced by 1
		/// 	hrows std::out_of_range if dimension or index is invalid
		[[nodiscard]] nd_span<const Ty, MaxRank> slice( size_type dim, size_type index ) const
		{
			if( dim >= rank_ )
			{
				throw std::out_of_range( "Dimension out of range" );
			}
			if( index >= extents_[dim] )
			{
				throw std::out_of_range( "Index out of bounds" );
			}

			std::array<size_type, MaxRank> new_extents;
			std::array<size_type, MaxRank> new_strides;

			size_type new_rank = rank_ - 1;
			size_type offset   = index * strides_[dim];

			size_type j = 0;
			for( size_t i = 0; i < rank_; ++i )
			{
				if( i != dim )
				{
					new_extents[j] = extents_[i];
					new_strides[j] = strides_[i];
					++j;
				}
			}

			for( size_t i = new_rank; i < MaxRank; ++i )
			{
				new_extents[i] = 0;
				new_strides[i] = 0;
			}

			return nd_span<const Ty, MaxRank>( data_.get( ) + offset, new_extents, new_strides, new_rank );
		}

		/// \brief Reshapes the array view (row-major contiguous)
		/// \param new_extents New shape extents
		/// \return Reshaped view
		[[nodiscard]] nd_span<Ty, MaxRank> reshape( std::initializer_list<size_type> new_extents )
		{
			return reshape_impl( new_extents.begin( ), new_extents.size( ) );
		}

		/// \brief Reshapes the array view (row-major contiguous)
		/// \param new_extents New shape extents
		/// \return Reshaped const view
		[[nodiscard]] nd_span<const Ty, MaxRank> reshape( std::initializer_list<size_type> new_extents ) const
		{
			return reshape_impl( new_extents.begin( ), new_extents.size( ) );
		}

		/// \brief Reshapes the array view with variadic extents
		/// \tparam Indices Variadic extent types
		/// \param new_extents New shape extents
		/// \return Reshaped view
		template<typename... Indices>
		[[nodiscard]] nd_span<Ty, MaxRank> reshape( Indices... new_extents )
		{
			size_type temp[] = { static_cast<size_type>( new_extents )... };
			return reshape_impl( temp, sizeof...( new_extents ) );
		}

		/// \brief Reshapes the array view with variadic extents (const)
		/// \tparam Indices Variadic extent types
		/// \param new_extents New shape extents
		/// \return Reshaped const view
		template<typename... Indices>
		[[nodiscard]] nd_span<const Ty, MaxRank> reshape( Indices... new_extents ) const
		{
			size_type temp[] = { static_cast<size_type>( new_extents )... };
			return reshape_impl( temp, sizeof...( new_extents ) );
		}

		/// \brief Returns a transposed view using an axis permutation
		/// \param axes Permutation of axes
		/// \return Transposed view
		[[nodiscard]] nd_span<Ty, MaxRank> transpose( std::initializer_list<size_type> axes )
		{
			return transpose_impl( axes.begin( ), axes.size( ), data_.get( ) );
		}

		/// \brief Returns a transposed view using an axis permutation (const)
		/// \param axes Permutation of axes
		/// \return Transposed const view
		[[nodiscard]] nd_span<const Ty, MaxRank> transpose( std::initializer_list<size_type> axes ) const
		{
			return transpose_impl( axes.begin( ), axes.size( ), data_.get( ) );
		}

		/// \brief Returns a transposed view by swapping the last two axes
		/// \return Transposed view
		[[nodiscard]] nd_span<Ty, MaxRank> T( )
		{
			size_type axes_rank = 0;
			auto axes = make_T_axes( axes_rank );
			return transpose_impl( axes.data( ), axes_rank, data_.get( ) );
		}

		/// \brief Returns a transposed view by swapping the last two axes (const)
		/// \return Transposed const view
		[[nodiscard]] nd_span<const Ty, MaxRank> T( ) const
		{
			size_type axes_rank = 0;
			auto axes = make_T_axes( axes_rank );
			return transpose_impl( axes.data( ), axes_rank, data_.get( ) );
		}

		/// \brief Flattens the array into a 1D view
		/// \return 1D view of the data
		[[nodiscard]] nd_span<Ty, MaxRank> flatten( ) { return reshape( size_ ); }

		/// \brief Flattens the array into a 1D view (const)
		/// \return 1D const view of the data
		[[nodiscard]] nd_span<const Ty, MaxRank> flatten( ) const { return reshape( size_ ); }

		/// \brief Removes dimensions of extent 1
		/// \return View with singleton dimensions removed
		[[nodiscard]] nd_span<Ty, MaxRank> squeeze( )
		{
			return squeeze_impl( data_.get( ) );
		}

		/// \brief Removes dimensions of extent 1 (const)
		/// \return Const view with singleton dimensions removed
		[[nodiscard]] nd_span<const Ty, MaxRank> squeeze( ) const
		{
			return squeeze_impl( data_.get( ) );
		}

		/// \brief Gets the size of a specific dimension
		/// \param dim Dimension index (0-based)
		/// \return Size of the specified dimension
		/// \throws std::out_of_range if dimension is >= rank
		[[nodiscard]] size_type extent( size_type dim ) const
		{
			if( dim >= rank_ )
			{
				throw std::out_of_range( "Dimension out of range" );
			}
			return extents_[dim];
		}

		/// \brief Gets the stride of a specific dimension
		/// \param dim Dimension index (0-based)
		/// \return Stride of the specified dimension
		/// \throws std::out_of_range if dimension is >= rank
		[[nodiscard]] size_type stride( size_type dim ) const
		{
			if( dim >= rank_ )
			{
				throw std::out_of_range( "Dimension out of range" );
			}
			return strides_[dim];
		}

		/// \brief Gets the active extents as a view sized to rank()
		/// \return View of extents for the active dimensions
		[[nodiscard]] detail::extents_view<size_type> extents( ) const noexcept { return { extents_.data( ), rank_ }; }

		/// \brief Gets the total number of elements in the array
		/// \return Total number of elements (product of all extents)
		[[nodiscard]] size_type size( ) const noexcept { return size_; }

		/// \brief Gets the number of dimensions
		/// \return Current rank (number of dimensions)
		[[nodiscard]] size_type rank( ) const noexcept { return rank_; }

		/// \brief Gets the maximum number of dimensions supported
		/// \return MaxRank template parameter
		[[nodiscard]] static constexpr size_type max_rank( ) noexcept { return MaxRank; }

		/// \brief Gets a pointer to the underlying data (non-const)
		/// \return Pointer to the first element
		[[nodiscard]] pointer data( ) noexcept { return data_.get( ); }

		/// \brief Gets a pointer to the underlying data (const)
		/// \return Const pointer to the first element
		[[nodiscard]] const_pointer data( ) const noexcept { return data_.get( ); }

		/// \brief Returns a pointer to the first element for flat iteration
		[[nodiscard]] pointer begin( ) noexcept { return data_.get( ); }

		/// \brief Returns a pointer past the last element for flat iteration
		[[nodiscard]] pointer end( ) noexcept { return data_.get( ) + size_; }

		/// \brief Returns a const pointer to the first element for flat iteration
		[[nodiscard]] const_pointer begin( ) const noexcept { return data_.get( ); }

		/// \brief Returns a const pointer past the last element for flat iteration
		[[nodiscard]] const_pointer end( ) const noexcept { return data_.get( ) + size_; }

		/// \brief Returns a const pointer to the first element for flat iteration
		[[nodiscard]] const_pointer cbegin( ) const noexcept { return data_.get( ); }

		/// \brief Returns a const pointer past the last element for flat iteration
		[[nodiscard]] const_pointer cend( ) const noexcept { return data_.get( ) + size_; }

		/// \brief Fills all elements with a specified value
		/// \param value Value to fill the array with
		/// \example
		/// \code
		/// nd_array<double> arr(3, 4);
		/// arr.fill(0.0);  // Set all elements to zero
		/// \endcode
		void fill( const Ty& value ) { std::fill( data_.get( ), data_.get( ) + size_, value ); }

		/// \brief Applies a function to each element
		/// \tparam Func Function type (typically a lambda or function object)
		/// \param func Function that takes a Ty and returns a Ty
		/// \example
		/// \code
		/// nd_array<double> arr(3, 4);
		/// arr.fill(1.0);
		/// arr.apply([](double x) { return x * 2.0; });  // Double all elements
		/// \endcode
		template<typename Func>
		void apply( Func&& func )
		{
			for( size_t i = 0; i < size_; ++i )
			{
				data_[i] = func( data_[i] );
			}
		}

	private:
		std::unique_ptr<Ty[]> data_;             ///< Owned data storage
		std::array<size_type, MaxRank> extents_; ///< Size of each dimension
		std::array<size_type, MaxRank> strides_; ///< Stride for each dimension
		size_type size_;                         ///< Total number of elements
		size_type rank_;                         ///< Actual number of dimensions

		/// \brief Helper function to create an nd_array from an nd_span
		template<typename U>
		static nd_array from_span_impl( const nd_span<U, MaxRank>& span )
		{
			static_assert( std::is_convertible<U, Ty>::value, "Span element type must be convertible" );

			nd_array result;
			if( span.rank( ) > MaxRank )
			{
				throw std::invalid_argument( "Rank exceeds MaxRank" );
			}

			result.rank_ = span.rank( );
			result.extents_.fill( 0 );
			result.strides_.fill( 0 );

			for( size_type i = 0; i < result.rank_; ++i )
			{
				result.extents_[i] = span.extent( i );
			}

			result.compute_strides( );
			result.size_ = detail::compute_size<MaxRank>( result.extents_, result.rank_ );
			if( result.size_ > 0 )
			{
				result.data_     = std::make_unique<Ty[]>( result.size_ );
				size_type offset = 0;
				for( const auto& value: span )
				{
					result.data_[offset++] = static_cast<Ty>( value );
				}
			}

			return result;
		}

		/// \brief Computes row-major strides from extents
		constexpr void compute_strides( ) noexcept { detail::stride_computer<MaxRank>::compute( strides_, extents_, rank_ ); }

		/// \brief Computes total number of elements from extents
		/// \return Product of all dimension sizes
		[[nodiscard]] constexpr size_type compute_size( ) const noexcept
		{
			if( rank_ == 0 )
				return 0;
			size_type s = 1;
			for( size_t i = 0; i < rank_; ++i )
			{
				s *= extents_[i];
			}
			return s;
		}

		[[nodiscard]] nd_span<Ty, MaxRank> reshape_impl( const size_type* new_extents, size_type new_rank )
		{
			if( new_rank > MaxRank )
			{
				throw std::invalid_argument( "Rank exceeds MaxRank" );
			}

			std::array<size_type, MaxRank> new_extents_array{};
			for( size_t i = 0; i < new_rank; ++i )
			{
				new_extents_array[i] = new_extents[i];
			}

			const size_type new_size = detail::compute_size<MaxRank>( new_extents_array, new_rank );
			if( new_size != size_ )
			{
				throw std::invalid_argument( "Reshape size mismatch" );
			}

			std::array<size_type, MaxRank> new_strides{};
			detail::stride_computer<MaxRank>::compute( new_strides, new_extents_array, new_rank );
			return nd_span<Ty, MaxRank>( data_.get( ), new_extents_array, new_strides, new_rank );
		}

		[[nodiscard]] nd_span<const Ty, MaxRank> reshape_impl( const size_type* new_extents, size_type new_rank ) const
		{
			if( new_rank > MaxRank )
			{
				throw std::invalid_argument( "Rank exceeds MaxRank" );
			}

			std::array<size_type, MaxRank> new_extents_array{};
			for( size_t i = 0; i < new_rank; ++i )
			{
				new_extents_array[i] = new_extents[i];
			}

			const size_type new_size = detail::compute_size<MaxRank>( new_extents_array, new_rank );
			if( new_size != size_ )
			{
				throw std::invalid_argument( "Reshape size mismatch" );
			}

			std::array<size_type, MaxRank> new_strides{};
			detail::stride_computer<MaxRank>::compute( new_strides, new_extents_array, new_rank );
			return nd_span<const Ty, MaxRank>( data_.get( ), new_extents_array, new_strides, new_rank );
		}

		template<typename PointerType>
		[[nodiscard]] auto transpose_impl( const size_type* axes, size_type axes_rank, PointerType data_ptr ) const
		{
			if( axes_rank != rank_ )
			{
				throw std::invalid_argument( "Permutation size must match rank" );
			}
			detail::validate_permutation<MaxRank>( axes, axes_rank );

			std::array<size_type, MaxRank> new_extents;
			std::array<size_type, MaxRank> new_strides;

			for( size_t i = 0; i < rank_; ++i )
			{
				new_extents[i] = extents_[axes[i]];
				new_strides[i] = strides_[axes[i]];
			}
			for( size_t i = rank_; i < MaxRank; ++i )
			{
				new_extents[i] = 0;
				new_strides[i] = 0;
			}

			using ViewType = nd_span<std::remove_pointer_t<PointerType>, MaxRank>;
			return ViewType( data_ptr, new_extents, new_strides, rank_ );
		}

		[[nodiscard]] std::array<size_type, MaxRank> make_T_axes( size_type& axes_rank ) const
		{
			axes_rank = rank_;
			std::array<size_type, MaxRank> axes{};
			for( size_t i = 0; i < rank_; ++i )
			{
				axes[i] = i;
			}
			if( rank_ >= 2 )
			{
				std::swap( axes[rank_ - 1], axes[rank_ - 2] );
			}
			return axes;
		}

		template<typename PointerType>
		[[nodiscard]] auto squeeze_impl( PointerType data_ptr ) const
		{
			std::array<size_type, MaxRank> new_extents;
			std::array<size_type, MaxRank> new_strides;
			size_type new_rank = 0;

			for( size_t i = 0; i < rank_; ++i )
			{
				if( extents_[i] != 1 )
				{
					new_extents[new_rank] = extents_[i];
					new_strides[new_rank] = strides_[i];
					++new_rank;
				}
			}

			for( size_t i = new_rank; i < MaxRank; ++i )
			{
				new_extents[i] = 0;
				new_strides[i] = 0;
			}

			using ViewType = nd_span<std::remove_pointer_t<PointerType>, MaxRank>;
			return ViewType( data_ptr, new_extents, new_strides, new_rank );
		}
	};

} // namespace cppa