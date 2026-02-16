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
			[[nodiscard]] static constexpr size_type compute( const std::array<size_type, MaxRank>& t_extents, const std::array<size_type, MaxRank>& t_strides,
			                                                  Indices... t_indices )
			{
				std::array<size_type, sizeof...( t_indices )> idx = { static_cast<size_type>( t_indices )... };
				size_type offset                                  = 0;
				for( size_t i = 0; i < sizeof...( t_indices ); ++i )
				{
					if( idx[i] >= t_extents[i] )
					{
						throw std::out_of_range( "Index out of bounds" );
					}
					offset += idx[i] * t_strides[i];
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
			static constexpr void compute( std::array<size_type, MaxRank>& t_strides, const std::array<size_type, MaxRank>& t_extents, size_type t_rank ) noexcept
			{
				if( t_rank == 0 )
					return;

				t_strides[t_rank - 1] = 1;
				for( size_t i = t_rank - 1; i > 0; --i )
				{
					t_strides[i - 1] = t_strides[i] * t_extents[i];
				}

				for( size_t i = t_rank; i < MaxRank; ++i )
				{
					t_strides[i] = 0;
				}
			}
		};

		template<typename SizeType>
		struct extents_view
		{
			const SizeType* data = nullptr;
			size_t size          = 0;

			[[nodiscard]] const SizeType* begin( ) const noexcept { return data; }
			[[nodiscard]] const SizeType* end( ) const noexcept { return data + size; }
		};

		/// \brief Computes total number of elements from extents
		/// \tparam MaxRank Maximum number of dimensions supported
		/// \param extents Extents array
		/// \param rank Number of active dimensions
		/// \return Product of active extents (0 when rank is 0)
		template<size_t MaxRank>
		[[nodiscard]] constexpr size_t compute_size( const std::array<size_t, MaxRank>& t_extents, size_t t_rank ) noexcept
		{
			if( t_rank == 0 )
				return 0;
			size_t s = 1;
			for( size_t i = 0; i < t_rank; ++i )
			{
				s *= t_extents[i];
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
		[[nodiscard]] constexpr bool is_contiguous( const std::array<size_t, MaxRank>& t_extents, const std::array<size_t, MaxRank>& t_strides, size_t t_rank ) noexcept
		{
			if( t_rank == 0 )
				return true;
			if( compute_size<MaxRank>( t_extents, t_rank ) == 0 )
				return true;
			if( t_strides[t_rank - 1] != 1 )
				return false;
			for( size_t i = t_rank - 1; i > 0; --i )
			{
				if( t_strides[i - 1] != t_strides[i] * t_extents[i] )
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
		inline void validate_permutation( const size_t* t_axes, size_t t_rank )
		{
			if( t_rank > MaxRank )
			{
				throw std::invalid_argument( "Permutation size must be <= MaxRank" );
			}
			std::array<bool, MaxRank> seen { };
			for( size_t i = 0; i < t_rank; ++i )
			{
				const size_t axis = t_axes[i];
				if( axis >= t_rank || seen[axis] )
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
		using size_type       = size_t;    ///< Type for sizes and indices
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
		constexpr nd_span( pointer t_data, const std::array<size_type, MaxRank>& t_extents, const std::array<size_type, MaxRank>& t_strides, size_type t_rank ) noexcept
		    : m_data( t_data )
		    , m_extents( t_extents )
		    , m_strides( t_strides )
		    , m_rank( t_rank )
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
		nd_span( pointer t_data, std::initializer_list<size_type> t_extents ) : m_data( t_data ), m_rank( t_extents.size( ) )
		{
			if( m_rank > MaxRank )
			{
				throw std::invalid_argument( "Rank exceeds MaxRank" );
			}

			size_t idx = 0;
			for( auto extent: t_extents )
			{
				m_extents[idx++] = extent;
			}
			for( size_t i = m_rank; i < MaxRank; ++i )
			{
				m_extents[i] = 0;
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
		nd_span( pointer t_data, const Container& t_extents, std::enable_if_t<!std::is_integral_v<Container> && !std::is_same_v<Container, nd_span>>* = nullptr )
		    : m_data( t_data )
		    , m_rank( t_extents.size( ) )
		{
			if( m_rank > MaxRank )
			{
				throw std::invalid_argument( "Rank exceeds MaxRank" );
			}

			size_t idx = 0;
			for( auto extent: t_extents )
			{
				m_extents[idx++] = static_cast<size_type>( extent );
			}
			for( size_t i = m_rank; i < MaxRank; ++i )
			{
				m_extents[i] = 0;
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
		nd_span( pointer t_data, Indices... t_indices ) : m_data( t_data )
		                                                , m_rank( sizeof...( t_indices ) )
		{
			static_assert( sizeof...( t_indices ) <= MaxRank, "Too many dimensions" );

			std::array<size_type, sizeof...( t_indices )> temp = { static_cast<size_type>( t_indices )... };
			for( size_t i = 0; i < m_rank; ++i )
			{
				m_extents[i] = temp[i];
			}
			for( size_t i = m_rank; i < MaxRank; ++i )
			{
				m_extents[i] = 0;
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
		[[nodiscard]] reference operator( )( Indices... t_indices )
		{
			return m_data[detail::offset_computer<MaxRank>::compute( m_extents, m_strides, t_indices... )];
		}

		/// \brief Accesses an element with multi-dimensional indexing (const)
		/// \tparam Indices Variadic index types (typically size_t)
		/// \param indices Multi-dimensional indices (i, j, k, ...)
		/// \return Const reference to the element at the specified location
		/// \throws std::out_of_range if any index is out of bounds
		template<typename... Indices>
		[[nodiscard]] const_reference operator( )( Indices... t_indices ) const
		{
			return data_[detail::offset_computer<MaxRank>::compute( extents_, strides_, indices... )];
		}

		/// \brief Creates a subspan by restricting a range along one dimension
		/// \param dim Dimension to restrict (0-based)
		/// \param range Pair of {start (inclusive), end (exclusive)} indices for that dimension
		/// \return New nd_span view of the restricted data
		/// \throws std::out_of_range if dimension or range is invalid
		/// \example
		/// \code
		/// nd_span<double> span(data, 5, 10);
		/// auto sub = span.subspan(0, 1, 4);  // Rows 1-3, all columns
		/// \endcode
		[[nodiscard]] nd_span subspan( size_type t_dim, std::pair<size_type, size_type> t_range ) const
		{
			size_type start = t_range.first;
			size_type end   = t_range.second;
			if( t_dim >= m_rank )
			{
				throw std::out_of_range( "Dimension out of range" );
			}
			if( start >= m_extents[t_dim] || end > m_extents[t_dim] || start >= end )
			{
				throw std::out_of_range( "Invalid range for subspan" );
			}

			std::array<size_type, MaxRank> new_extents = m_extents;
			new_extents[t_dim]                         = end - start;

			size_type offset = start * m_strides[t_dim];

			return nd_span( m_data + offset, new_extents, m_strides, m_rank );
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
		[[nodiscard]] nd_span slice( size_type t_dim, size_type t_index ) const
		{
			if( t_dim >= m_rank )
			{
				throw std::out_of_range( "Dimension out of range" );
			}
			if( t_index >= m_extents[t_dim] )
			{
				throw std::out_of_range( "Index out of bounds" );
			}

			std::array<size_type, MaxRank> new_extents;
			std::array<size_type, MaxRank> new_strides;

			size_type new_rank = m_rank - 1;
			size_type offset   = t_index * m_strides[t_dim];

			size_type j = 0;
			for( size_t i = 0; i < m_rank; ++i )
			{
				if( i != t_dim )
				{
					new_extents[j] = m_extents[i];
					new_strides[j] = m_strides[i];
					++j;
				}
			}

			for( size_t i = new_rank; i < MaxRank; ++i )
			{
				new_extents[i] = 0;
				new_strides[i] = 0;
			}

			return nd_span( m_data + offset, new_extents, new_strides, new_rank );
		}

		/// \brief Reshapes the span (view-only, row-major contiguous required)
		/// \param new_extents New shape extents
		/// \return Reshaped view
		/// \throws std::invalid_argument if rank exceeds MaxRank or size mismatch
		/// \throws std::runtime_error if the view is not contiguous
		[[nodiscard]] nd_span reshape( std::initializer_list<size_type> t_new_extents ) const { return reshape_impl( t_new_extents.begin( ), t_new_extents.size( ) ); }

		/// \brief Reshapes the span with variadic extents (view-only, row-major contiguous required)
		/// \tparam Indices Variadic extent types
		/// \param new_extents New shape extents
		/// \return Reshaped view
		template<typename... Indices>
		[[nodiscard]] nd_span reshape( Indices... t_new_extents ) const
		{
			std::array<size_type, sizeof...( t_new_extents )> temp = { static_cast<size_type>( t_new_extents )... };
			return reshape_impl( temp.data( ), sizeof...( t_new_extents ) );
		}

		/// \brief Returns a transposed view using an axis permutation
		/// \param axes Permutation of axes
		/// \return Transposed view
		/// 	hrows std::invalid_argument if permutation is invalid
		[[nodiscard]] nd_span transpose( std::initializer_list<size_type> t_axes ) const { return transpose_impl( t_axes.begin( ), t_axes.size( ) ); }

		/// \brief Returns a transposed view by swapping the last two axes
		/// \return Transposed view
		[[nodiscard]] nd_span T( ) const // NOLINT(readability-identifier-naming)
		{
			size_type axes_rank = 0;
			auto axes           = make_t_axes( axes_rank );
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

			for( size_t i = 0; i < m_rank; ++i )
			{
				if( m_extents[i] != 1 )
				{
					new_extents[new_rank] = m_extents[i];
					new_strides[new_rank] = m_strides[i];
					++new_rank;
				}
			}

			for( size_t i = new_rank; i < MaxRank; ++i )
			{
				new_extents[i] = 0;
				new_strides[i] = 0;
			}

			return nd_span( m_data, new_extents, new_strides, new_rank );
		}

		/// \brief Gets the size of a specific dimension
		/// \param dim Dimension index (0-based)
		/// \return Size of the specified dimension
		/// \throws std::out_of_range if dimension is >= rank
		[[nodiscard]] size_type extent( size_type t_dim ) const
		{
			if( t_dim >= m_rank )
			{
				throw std::out_of_range( "Dimension out of range" );
			}
			return m_extents[t_dim];
		}

		/// \brief Gets the stride of a specific dimension
		/// \param dim Dimension index (0-based)
		/// \return Stride of the specified dimension
		/// \throws std::out_of_range if dimension is >= rank
		[[nodiscard]] size_type stride( size_type t_dim ) const
		{
			if( t_dim >= m_rank )
			{
				throw std::out_of_range( "Dimension out of range" );
			}
			return m_strides[t_dim];
		}

		/// \brief Gets the active extents as a view sized to rank()
		/// \return View of extents for the active dimensions
		[[nodiscard]] detail::extents_view<size_type> extents( ) const noexcept { return { m_extents.data( ), m_rank }; }

		/// \brief Gets the total number of elements in the span
		/// \return Total number of elements (product of all extents)
		[[nodiscard]] size_type size( ) const noexcept { return compute_size( ); }

		/// \brief Gets the number of dimensions
		/// \return Current rank (number of dimensions)
		[[nodiscard]] size_type rank( ) const noexcept { return m_rank; }

		/// \brief Gets the maximum number of dimensions supported
		/// \return MaxRank template parameter
		[[nodiscard]] static constexpr size_type max_rank( ) noexcept { return MaxRank; }

		/// \brief Gets a pointer to the underlying data (non-const)
		/// \return Pointer to the first element
		[[nodiscard]] pointer data( ) noexcept { return m_data; }

		/// \brief Gets a pointer to the underlying data (const)
		/// \return Const pointer to the first element
		[[nodiscard]] const_pointer data( ) const noexcept { return m_data; }

		/// \brief Returns a pointer to the first element for flat iteration
		[[nodiscard]] pointer begin( ) noexcept { return m_data; }

		/// \brief Returns a pointer past the last element for flat iteration
		[[nodiscard]] pointer end( ) noexcept { return m_data + size( ); }

		/// \brief Returns a const pointer to the first element for flat iteration
		[[nodiscard]] const_pointer begin( ) const noexcept { return m_data; }

		/// \brief Returns a const pointer past the last element for flat iteration
		[[nodiscard]] const_pointer end( ) const noexcept { return m_data + size( ); }

		/// \brief Returns a const pointer to the first element for flat iteration
		[[nodiscard]] const_pointer cbegin( ) const noexcept { return m_data; }

		/// \brief Returns a const pointer past the last element for flat iteration
		[[nodiscard]] const_pointer cend( ) const noexcept { return m_data + size( ); }

	private:
		pointer m_data;                           ///< Pointer to the first element
		std::array<size_type, MaxRank> m_extents; ///< Size of each dimension
		std::array<size_type, MaxRank> m_strides; ///< Stride for each dimension
		size_type m_rank;                         ///< Actual number of dimensions

		/// \brief Computes row-major strides from extents
		constexpr void compute_strides( ) noexcept { detail::stride_computer<MaxRank>::compute( m_strides, m_extents, m_rank ); }

		/// \brief Computes total number of elements from extents
		[[nodiscard]] constexpr size_type compute_size( ) const noexcept
		{
			if( m_rank == 0 )
				return 0;
			size_type s = 1;
			for( size_t i = 0; i < m_rank; ++i )
			{
				s *= m_extents[i];
			}
			return s;
		}

		[[nodiscard]] nd_span reshape_impl( const size_type* t_new_extents, size_type t_new_rank ) const
		{
			if( t_new_rank > MaxRank )
			{
				throw std::invalid_argument( "Rank exceeds MaxRank" );
			}
			if( !detail::is_contiguous<MaxRank>( m_extents, m_strides, m_rank ) )
			{
				throw std::runtime_error( "Reshape requires contiguous data" );
			}

			std::array<size_type, MaxRank> new_extents_array { };
			for( size_t i = 0; i < t_new_rank; ++i )
			{
				new_extents_array[i] = t_new_extents[i];
			}

			const size_type new_size = detail::compute_size<MaxRank>( new_extents_array, t_new_rank );
			if( new_size != size( ) )
			{
				throw std::invalid_argument( "Reshape size mismatch" );
			}

			std::array<size_type, MaxRank> new_strides { };
			detail::stride_computer<MaxRank>::compute( new_strides, new_extents_array, t_new_rank );

			return nd_span( m_data, new_extents_array, new_strides, t_new_rank );
		}

		[[nodiscard]] nd_span transpose_impl( const size_type* t_axes, size_type t_axes_rank ) const
		{
			if( t_axes_rank != m_rank )
			{
				throw std::invalid_argument( "Permutation size must match rank" );
			}
			detail::validate_permutation<MaxRank>( t_axes, t_axes_rank );

			std::array<size_type, MaxRank> new_extents;
			std::array<size_type, MaxRank> new_strides;
			for( size_t i = 0; i < m_rank; ++i )
			{
				new_extents[i] = m_extents[t_axes[i]];
				new_strides[i] = m_strides[t_axes[i]];
			}
			for( size_t i = m_rank; i < MaxRank; ++i )
			{
				new_extents[i] = 0;
				new_strides[i] = 0;
			}

			return nd_span( m_data, new_extents, new_strides, m_rank );
		}

		[[nodiscard]] std::array<size_type, MaxRank> make_t_axes( size_type& t_axes_rank ) const
		{
			t_axes_rank = m_rank;
			std::array<size_type, MaxRank> axes { };
			for( size_t i = 0; i < m_rank; ++i )
			{
				axes[i] = i;
			}
			if( m_rank >= 2 )
			{
				std::swap( axes[m_rank - 1], axes[m_rank - 2] );
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
		using size_type       = size_t;    ///< Type for sizes and indices
		using reference       = Ty&;       ///< Reference to element
		using const_reference = const Ty&; ///< Const reference to element
		using pointer         = Ty*;       ///< Pointer to element
		using const_pointer   = const Ty*; ///< Const pointer to element

		/// \brief Constructs an empty array with no dimensions
		/// \note No memory is allocated
		nd_array( ) noexcept : m_data( nullptr ), m_size( 0 ), m_rank( 0 )
		{
			m_extents.fill( 0 );
			m_strides.fill( 0 );
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
		nd_array( std::initializer_list<size_type> t_extents ) : m_rank( t_extents.size( ) )
		{
			if( m_rank > MaxRank )
			{
				throw std::invalid_argument( "Rank exceeds MaxRank" );
			}

			size_t idx = 0;
			for( auto extent: t_extents )
			{
				m_extents[idx++] = extent;
			}
			for( size_t i = m_rank; i < MaxRank; ++i )
			{
				m_extents[i] = 0;
			}

			compute_strides( );
			m_size = compute_size( );
			m_data = std::make_unique<Ty[]>( m_size ); // NOLINT(modernize-avoid-c-arrays)
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
		nd_array( const Container& t_extents, std::enable_if_t<!std::is_integral_v<Container> && !std::is_same_v<Container, nd_array>>* = nullptr )
		    : m_rank( t_extents.size( ) )
		{
			if( m_rank > MaxRank )
			{
				throw std::invalid_argument( "Rank exceeds MaxRank" );
			}

			size_t idx = 0;
			for( auto extent: t_extents )
			{
				m_extents[idx++] = static_cast<size_type>( extent );
			}
			for( size_t i = m_rank; i < MaxRank; ++i )
			{
				m_extents[i] = 0;
			}

			compute_strides( );
			m_size = compute_size( );
			m_data = std::make_unique<Ty[]>( m_size ); // NOLINT(modernize-avoid-c-arrays)
		}

		/// \brief Constructs an array with variadic dimension sizes
		/// \tparam Indices Variadic index types (typically size_t or convertible to size_t)
		/// \param indices Dimension sizes as separate arguments
		/// \example
		/// \code
		/// nd_array<double> arr(2, 3, 4);  // 2x3x4 array
		/// \endcode
		template<typename... Indices>
		nd_array( Indices... t_indices ) : m_rank( sizeof...( t_indices ) )
		{
			static_assert( sizeof...( t_indices ) <= MaxRank, "Too many dimensions" );

			std::array<size_type, sizeof...( t_indices )> temp = { static_cast<size_type>( t_indices )... };
			for( size_t i = 0; i < m_rank; ++i )
			{
				m_extents[i] = temp[i];
			}
			for( size_t i = m_rank; i < MaxRank; ++i )
			{
				m_extents[i] = 0;
			}

			compute_strides( );
			m_size = compute_size( );
			m_data = std::make_unique<Ty[]>( m_size ); // NOLINT(modernize-avoid-c-arrays)
		}

		/// \brief Copy constructor - performs deep copy of data
		/// \param other Array to copy from
		nd_array( const nd_array& t_other ) : m_extents( t_other.m_extents ), m_strides( t_other.m_strides ), m_size( t_other.m_size ), m_rank( t_other.m_rank )
		{
			if( m_size > 0 )
			{
				m_data = std::make_unique<Ty[]>( m_size ); // NOLINT(modernize-avoid-c-arrays)
				std::copy( t_other.m_data.get( ), t_other.m_data.get( ) + m_size, m_data.get( ) );
			}
		}

		/// \brief Move constructor - transfers ownership of data
		nd_array( nd_array&& t_other ) noexcept = default;

		/// \brief Constructs an owning array by deep-copying an nd_span
		/// \param span Source span to copy
		/// 	hrows std::invalid_argument if span rank exceeds MaxRank
		explicit nd_array( const nd_span<const Ty, MaxRank>& t_span ) : nd_array( from_span( span ) ) {}

		/// \brief Constructs an owning array by deep-copying an nd_span
		/// \param span Source span to copy
		explicit nd_array( const nd_span<Ty, MaxRank>& t_span ) : nd_array( from_span( t_span ) ) {}

		/// \brief Copy assignment operator - performs deep copy of data
		/// \param other Array to copy from
		/// \return Reference to this array
		nd_array& operator=( const nd_array& t_other )
		{
			if( this != &t_other )
			{
				m_rank    = t_other.m_rank;
				m_size    = t_other.m_size;
				m_extents = t_other.m_extents;
				m_strides = t_other.m_strides;
				if( m_size > 0 )
				{
					m_data = std::make_unique<Ty[]>( m_size ); // NOLINT(modernize-avoid-c-arrays)
					std::copy( t_other.m_data.get( ), t_other.m_data.get( ) + m_size, m_data.get( ) );
				}
				else
				{
					m_data.reset( );
				}
			}
			return *this;
		}

		/// \brief Move assignment operator - transfers ownership of data
		/// \param other Array to move from
		/// \return Reference to this array
		nd_array& operator=( nd_array&& t_other ) noexcept = default;

		/// \brief Assigns from an nd_span by deep-copying its contents
		/// \param span Source span to copy
		/// \return Reference to this array
		nd_array& operator=( const nd_span<const Ty, MaxRank>& t_span ) { return *this = from_span( span ); }

		/// \brief Assigns from an nd_span by deep-copying its contents
		/// \param span Source span to copy
		/// \return Reference to this array
		nd_array& operator=( const nd_span<Ty, MaxRank>& t_span ) { return *this = from_span( t_span ); }

		/// \brief Creates an owning array by deep-copying an nd_span
		/// \param span Source span to copy
		/// \return Newly allocated array with the same contents
		/// 	hrows std::invalid_argument if span rank exceeds MaxRank
		static nd_array from_span( const nd_span<const Ty, MaxRank>& t_span ) { return from_span_impl( span ); }

		/// \brief Creates an owning array by deep-copying an nd_span
		/// \param span Source span to copy
		/// \return Newly allocated array with the same contents
		static nd_array from_span( const nd_span<Ty, MaxRank>& t_span ) { return from_span_impl( t_span ); }

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
		[[nodiscard]] reference operator( )( Indices... t_indices )
		{
			static_assert( sizeof...( t_indices ) <= MaxRank, "Too many indices" );
			return m_data[detail::offset_computer<MaxRank>::compute( m_extents, m_strides, t_indices... )];
		}

		/// \brief Accesses an element with multi-dimensional indexing (const)
		/// \tparam Indices Variadic index types (typically size_t)
		/// \param indices Multi-dimensional indices (i, j, k, ...)
		/// \return Const reference to the element at the specified location
		/// \throws std::out_of_range if any index is out of bounds
		template<typename... Indices>
		[[nodiscard]] const_reference operator( )( Indices... t_indices ) const
		{
			static_assert( sizeof...( t_indices ) <= MaxRank, "Too many indices" );
			return m_data[detail::offset_computer<MaxRank>::compute( m_extents, m_strides, t_indices... )];
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
		[[nodiscard]] nd_span<Ty, MaxRank> subspan( std::initializer_list<std::pair<size_type, size_type>> t_ranges )
		{
			std::array<size_type, MaxRank> new_extents = m_extents;
			std::array<size_type, MaxRank> new_strides = m_strides;
			size_type offset                           = 0;
			size_type dim                              = 0;

			for( const auto& [start, end]: t_ranges )
			{
				if( dim >= m_rank )
				{
					throw std::out_of_range( "Too many dimensions in subspan" );
				}
				if( start >= m_extents[dim] || end > m_extents[dim] || start >= end )
				{
					throw std::out_of_range( "Invalid range for subspan" );
				}
				offset += start * m_strides[dim];
				new_extents[dim] = end - start;
				++dim;
			}

			return nd_span<Ty, MaxRank>( m_data.get( ) + offset, new_extents, new_strides, m_rank );
		}

		/// \brief Creates a subspan with multiple dimension ranges (const)
		/// \param ranges Initializer list of {start, end} pairs for each dimension
		/// \return Non-owning view (nd_span) of the restricted data
		/// 	hrows std::out_of_range if too many dimensions or invalid ranges
		[[nodiscard]] nd_span<const Ty, MaxRank> subspan( std::initializer_list<std::pair<size_type, size_type>> t_ranges ) const
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
		[[nodiscard]] nd_span<Ty, MaxRank> subspan( size_type t_dim, size_type t_start, size_type t_end )
		{
			if( t_dim >= m_rank )
			{
				throw std::out_of_range( "Dimension out of range" );
			}
			if( t_start >= m_extents[t_dim] || t_end > m_extents[t_dim] || t_start >= t_end )
			{
				throw std::out_of_range( "Invalid range for subspan" );
			}

			std::array<size_type, MaxRank> new_extents = m_extents;
			new_extents[t_dim]                         = t_end - t_start;

			size_type offset = t_start * m_strides[t_dim];

			return nd_span<Ty, MaxRank>( m_data.get( ) + offset, new_extents, m_strides, m_rank );
		}

		/// \brief Creates a subspan by restricting a range along one dimension (const)
		/// \param dim Dimension to restrict (0-based)
		/// \param range Pair of {start (inclusive), end (exclusive)} indices for that dimension
		/// \return Non-owning view (nd_span) of the restricted data
		/// 	hrows std::out_of_range if dimension or range is invalid
		[[nodiscard]] nd_span<const Ty, MaxRank> subspan( size_type t_dim, std::pair<size_type, size_type> t_range ) const
		{
			size_type start = range.first;
			size_type end   = range.second;
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
		[[nodiscard]] nd_span<Ty, MaxRank> slice( size_type t_dim, size_type t_index )
		{
			if( t_dim >= m_rank )
			{
				throw std::out_of_range( "Dimension out of range" );
			}
			if( t_index >= m_extents[t_dim] )
			{
				throw std::out_of_range( "Index out of bounds" );
			}

			std::array<size_type, MaxRank> new_extents;
			std::array<size_type, MaxRank> new_strides;

			size_type new_rank = m_rank - 1;
			size_type offset   = t_index * m_strides[t_dim];

			size_type j = 0;
			for( size_t i = 0; i < m_rank; ++i )
			{
				if( i != t_dim )
				{
					new_extents[j] = m_extents[i];
					new_strides[j] = m_strides[i];
					++j;
				}
			}

			for( size_t i = new_rank; i < MaxRank; ++i )
			{
				new_extents[i] = 0;
				new_strides[i] = 0;
			}

			return nd_span<Ty, MaxRank>( m_data.get( ) + offset, new_extents, new_strides, new_rank );
		}

		/// \brief Creates a lower-dimensional view by fixing one dimension's index (const)
		/// \param dim Dimension to slice (0-based)
		/// \param index Index value to fix for that dimension
		/// \return Non-owning view (nd_span) with rank reduced by 1
		/// 	hrows std::out_of_range if dimension or index is invalid
		[[nodiscard]] nd_span<const Ty, MaxRank> slice( size_type t_dim, size_type t_index ) const
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
		[[nodiscard]] nd_span<Ty, MaxRank> reshape( std::initializer_list<size_type> t_new_extents ) { return reshape_impl( new_extents.begin( ), new_extents.size( ) ); }

		/// \brief Reshapes the array view (row-major contiguous)
		/// \param new_extents New shape extents
		/// \return Reshaped const view
		[[nodiscard]] nd_span<const Ty, MaxRank> reshape( std::initializer_list<size_type> t_new_extents ) const
		{
			return reshape_impl( new_extents.begin( ), new_extents.size( ) );
		}

		/// \brief Reshapes the array view with variadic extents
		/// \tparam Indices Variadic extent types
		/// \param new_extents New shape extents
		/// \return Reshaped view
		template<typename... Indices>
		[[nodiscard]] nd_span<Ty, MaxRank> reshape( Indices... t_new_extents )
		{
			std::array<size_type, sizeof...( t_new_extents )> temp = { static_cast<size_type>( t_new_extents )... };
			return reshape_impl( temp.data( ), sizeof...( t_new_extents ) );
		}

		/// \brief Reshapes the array view with variadic extents (const)
		/// \tparam Indices Variadic extent types
		/// \param new_extents New shape extents
		/// \return Reshaped const view
		template<typename... Indices>
		[[nodiscard]] nd_span<const Ty, MaxRank> reshape( Indices... t_new_extents ) const
		{
			std::array<size_type, sizeof...( t_new_extents )> temp = { static_cast<size_type>( t_new_extents )... };
			return reshape_impl( temp.data( ), sizeof...( t_new_extents ) );
		}

		/// \brief Returns a transposed view using an axis permutation
		/// \param axes Permutation of axes
		/// \return Transposed view
		[[nodiscard]] nd_span<Ty, MaxRank> transpose( std::initializer_list<size_type> t_axes )
		{
			return transpose_impl( t_axes.begin( ), t_axes.size( ), m_data.get( ) );
		}

		/// \brief Returns a transposed view using an axis permutation (const)
		/// \param axes Permutation of axes
		/// \return Transposed const view
		[[nodiscard]] nd_span<const Ty, MaxRank> transpose( std::initializer_list<size_type> t_axes ) const
		{
			return transpose_impl( t_axes.begin( ), t_axes.size( ), m_data.get( ) );
		}

		/// \brief Returns a transposed view by swapping the last two axes
		/// \return Transposed view
		[[nodiscard]] nd_span<Ty, MaxRank> T( ) // NOLINT(readability-identifier-naming)
		{
			size_type axes_rank = 0;
			auto axes           = make_t_axes( axes_rank );
			return transpose_impl( axes.data( ), axes_rank, m_data.get( ) );
		}

		/// \brief Returns a transposed view by swapping the last two axes (const)
		/// \return Transposed const view
		[[nodiscard]] nd_span<const Ty, MaxRank> T( ) const // NOLINT(readability-identifier-naming)
		{
			size_type axes_rank = 0;
			auto axes           = make_T_axes( axes_rank );
			return transpose_impl( axes.data( ), axes_rank, data_.get( ) );
		}

		/// \brief Flattens the array into a 1D view
		/// \return 1D view of the data
		[[nodiscard]] nd_span<Ty, MaxRank> flatten( ) { return reshape( m_size ); }

		/// \brief Flattens the array into a 1D view (const)
		/// \return 1D const view of the data
		[[nodiscard]] nd_span<const Ty, MaxRank> flatten( ) const { return reshape( m_size ); }

		/// \brief Removes dimensions of extent 1
		/// \return View with singleton dimensions removed
		[[nodiscard]] nd_span<Ty, MaxRank> squeeze( ) { return squeeze_impl( m_data.get( ) ); }

		/// \brief Removes dimensions of extent 1 (const)
		/// \return Const view with singleton dimensions removed
		[[nodiscard]] nd_span<const Ty, MaxRank> squeeze( ) const { return squeeze_impl( data_.get( ) ); }

		/// \brief Gets the size of a specific dimension
		/// \param dim Dimension index (0-based)
		/// \return Size of the specified dimension
		/// \throws std::out_of_range if dimension is >= rank
		[[nodiscard]] size_type extent( size_type t_dim ) const
		{
			if( t_dim >= m_rank )
			{
				throw std::out_of_range( "Dimension out of range" );
			}
			return m_extents[t_dim];
		}

		/// \brief Gets the stride of a specific dimension
		/// \param dim Dimension index (0-based)
		/// \return Stride of the specified dimension
		/// \throws std::out_of_range if dimension is >= rank
		[[nodiscard]] size_type stride( size_type t_dim ) const
		{
			if( t_dim >= m_rank )
			{
				throw std::out_of_range( "Dimension out of range" );
			}
			return m_strides[t_dim];
		}

		/// \brief Gets the active extents as a view sized to rank()
		/// \return View of extents for the active dimensions
		[[nodiscard]] detail::extents_view<size_type> extents( ) const noexcept { return { m_extents.data( ), m_rank }; }

		/// \brief Gets the total number of elements in the array
		/// \return Total number of elements (product of all extents)
		[[nodiscard]] size_type size( ) const noexcept { return m_size; }

		/// \brief Gets the number of dimensions
		/// \return Current rank (number of dimensions)
		[[nodiscard]] size_type rank( ) const noexcept { return m_rank; }

		/// \brief Gets the maximum number of dimensions supported
		/// \return MaxRank template parameter
		[[nodiscard]] static constexpr size_type max_rank( ) noexcept { return MaxRank; }

		/// \brief Gets a pointer to the underlying data (non-const)
		/// \return Pointer to the first element
		[[nodiscard]] pointer data( ) noexcept { return m_data.get( ); }

		/// \brief Gets a pointer to the underlying data (const)
		/// \return Const pointer to the first element
		[[nodiscard]] const_pointer data( ) const noexcept { return data_.get( ); }

		/// \brief Returns a pointer to the first element for flat iteration
		[[nodiscard]] pointer begin( ) noexcept { return m_data.get( ); }

		/// \brief Returns a pointer past the last element for flat iteration
		[[nodiscard]] pointer end( ) noexcept { return m_data.get( ) + m_size; }

		/// \brief Returns a const pointer to the first element for flat iteration
		[[nodiscard]] const_pointer begin( ) const noexcept { return m_data.get( ); }

		/// \brief Returns a const pointer past the last element for flat iteration
		[[nodiscard]] const_pointer end( ) const noexcept { return m_data.get( ) + m_size; }

		/// \brief Returns a const pointer to the first element for flat iteration
		[[nodiscard]] const_pointer cbegin( ) const noexcept { return m_data.get( ); }

		/// \brief Returns a const pointer past the last element for flat iteration
		[[nodiscard]] const_pointer cend( ) const noexcept { return data_.get( ) + size_; }

		/// \brief Fills all elements with a specified value
		/// \param value Value to fill the array with
		/// \example
		/// \code
		/// nd_array<double> arr(3, 4);
		/// arr.fill(0.0);  // Set all elements to zero
		/// \endcode
		void fill( const Ty& t_value ) { std::fill( m_data.get( ), m_data.get( ) + m_size, t_value ); }

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
		void apply( Func&& t_func )
		{
			for( size_t i = 0; i < m_size; ++i )
			{
				m_data[i] = t_func( m_data[i] );
			}
		}

	private:
		/// \brief Internal owned data storage
		std::unique_ptr<Ty[]> m_data;             // NOLINT(modernize-avoid-c-arrays)
		std::array<size_type, MaxRank> m_extents; ///< Size of each dimension
		std::array<size_type, MaxRank> m_strides; ///< Stride for each dimension
		size_type m_size;                         ///< Total number of elements
		size_type m_rank;                         ///< Actual number of dimensions

		/// \brief Helper function to create an nd_array from an nd_span
		template<typename U>
		static nd_array from_span_impl( const nd_span<U, MaxRank>& t_span )
		{
			static_assert( std::is_convertible_v<U, Ty>, "Span element type must be convertible" );

			nd_array result;
			if( t_span.rank( ) > MaxRank )
			{
				throw std::invalid_argument( "Rank exceeds MaxRank" );
			}

			result.m_rank = t_span.rank( );
			result.m_extents.fill( 0 );
			result.m_strides.fill( 0 );

			for( size_type i = 0; i < result.m_rank; ++i )
			{
				result.m_extents[i] = t_span.extent( i );
			}

			result.compute_strides( );
			result.m_size = detail::compute_size<MaxRank>( result.m_extents, result.m_rank );
			if( result.m_size > 0 )
			{
				result.m_data    = std::make_unique<Ty[]>( result.m_size ); // NOLINT(modernize-avoid-c-arrays)
				size_type offset = 0;
				for( const auto& value: t_span )
				{
					result.m_data[offset++] = static_cast<Ty>( value );
				}
			}

			return result;
		}

		/// \brief Computes row-major strides from extents
		constexpr void compute_strides( ) noexcept { detail::stride_computer<MaxRank>::compute( m_strides, m_extents, m_rank ); }

		/// \brief Computes total number of elements from extents
		/// \return Product of all dimension sizes
		[[nodiscard]] constexpr size_type compute_size( ) const noexcept
		{
			if( m_rank == 0 )
				return 0;
			size_type s = 1;
			for( size_t i = 0; i < m_rank; ++i )
			{
				s *= m_extents[i];
			}
			return s;
		}

		[[nodiscard]] nd_span<Ty, MaxRank> reshape_impl( const size_type* t_new_extents, size_type t_new_rank )
		{
			if( t_new_rank > MaxRank )
			{
				throw std::invalid_argument( "Rank exceeds MaxRank" );
			}

			std::array<size_type, MaxRank> new_extents_array { };
			for( size_t i = 0; i < t_new_rank; ++i )
			{
				new_extents_array[i] = t_new_extents[i];
			}

			const size_type new_size = detail::compute_size<MaxRank>( new_extents_array, t_new_rank );
			if( new_size != m_size )
			{
				throw std::invalid_argument( "Reshape size mismatch" );
			}

			std::array<size_type, MaxRank> new_strides { };
			detail::stride_computer<MaxRank>::compute( new_strides, new_extents_array, t_new_rank );
			return nd_span<Ty, MaxRank>( m_data.get( ), new_extents_array, new_strides, t_new_rank );
		}

		[[nodiscard]] nd_span<const Ty, MaxRank> reshape_impl( const size_type* t_new_extents, size_type t_new_rank ) const
		{
			if( t_new_rank > MaxRank )
			{
				throw std::invalid_argument( "Rank exceeds MaxRank" );
			}

			std::array<size_type, MaxRank> new_extents_array { };
			for( size_t i = 0; i < t_new_rank; ++i )
			{
				new_extents_array[i] = t_new_extents[i];
			}

			const size_type new_size = detail::compute_size<MaxRank>( new_extents_array, t_new_rank );
			if( new_size != m_size )
			{
				throw std::invalid_argument( "Reshape size mismatch" );
			}

			std::array<size_type, MaxRank> new_strides { };
			detail::stride_computer<MaxRank>::compute( new_strides, new_extents_array, t_new_rank );
			return nd_span<const Ty, MaxRank>( m_data.get( ), new_extents_array, new_strides, t_new_rank );
		}

		template<typename PointerType>
		[[nodiscard]] auto transpose_impl( const size_type* t_axes, size_type t_axes_rank, PointerType t_data_ptr ) const
		{
			if( t_axes_rank != m_rank )
			{
				throw std::invalid_argument( "Permutation size must match rank" );
			}
			detail::validate_permutation<MaxRank>( t_axes, t_axes_rank );

			std::array<size_type, MaxRank> new_extents;
			std::array<size_type, MaxRank> new_strides;

			for( size_t i = 0; i < m_rank; ++i )
			{
				new_extents[i] = m_extents[t_axes[i]];
				new_strides[i] = m_strides[t_axes[i]];
			}
			for( size_t i = m_rank; i < MaxRank; ++i )
			{
				new_extents[i] = 0;
				new_strides[i] = 0;
			}

			using ViewType = nd_span<std::remove_pointer_t<PointerType>, MaxRank>;
			return ViewType( t_data_ptr, new_extents, new_strides, m_rank );
		}

		[[nodiscard]] std::array<size_type, MaxRank> make_t_axes( size_type& t_axes_rank ) const
		{
			t_axes_rank = m_rank;
			std::array<size_type, MaxRank> axes { };
			for( size_t i = 0; i < m_rank; ++i )
			{
				axes[i] = i;
			}
			if( m_rank >= 2 )
			{
				std::swap( axes[m_rank - 1], axes[m_rank - 2] );
			}
			return axes;
		}

		template<typename PointerType>
		[[nodiscard]] auto squeeze_impl( PointerType t_data_ptr ) const
		{
			std::array<size_type, MaxRank> new_extents;
			std::array<size_type, MaxRank> new_strides;
			size_type new_rank = 0;

			for( size_t i = 0; i < m_rank; ++i )
			{
				if( m_extents[i] != 1 )
				{
					new_extents[new_rank] = m_extents[i];
					new_strides[new_rank] = m_strides[i];
					++new_rank;
				}
			}

			for( size_t i = new_rank; i < MaxRank; ++i )
			{
				new_extents[i] = 0;
				new_strides[i] = 0;
			}

			using ViewType = nd_span<std::remove_pointer_t<PointerType>, MaxRank>;
			return ViewType( t_data_ptr, new_extents, new_strides, new_rank );
		}
	};

} // namespace cppa