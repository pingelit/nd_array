#pragma once

#include <algorithm>
#include <array>
#include <initializer_list>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <type_traits>


template<typename T, size_t MaxRank = 8>
class nd_array
{
public:
	using value_type      = T;
	using size_type       = size_t;
	using reference       = T&;
	using const_reference = const T&;
	using pointer         = T*;
	using const_pointer   = const T*;

	nd_array( ) : data_( nullptr ), size_( 0 ), rank_( 0 )
	{
		extents_.fill( 0 );
		strides_.fill( 0 );
	}

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
		data_ = std::make_unique<T[]>( size_ );
	}

	template<typename Container>
	nd_array( const Container& extents,
	          typename std::enable_if<
	              !std::is_integral<Container>::value &&
	              !std::is_same<Container, nd_array>::value>::type* = nullptr )
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
		data_ = std::make_unique<T[]>( size_ );
	}

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
		data_ = std::make_unique<T[]>( size_ );
	}

	nd_array( const nd_array& other ) : rank_( other.rank_ ), size_( other.size_ ), extents_( other.extents_ ), strides_( other.strides_ )
	{
		if( size_ > 0 )
		{
			data_ = std::make_unique<T[]>( size_ );
			std::copy( other.data_.get( ), other.data_.get( ) + size_, data_.get( ) );
		}
	}

	nd_array( nd_array&& other ) noexcept = default;

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
				data_ = std::make_unique<T[]>( size_ );
				std::copy( other.data_.get( ), other.data_.get( ) + size_, data_.get( ) );
			}
			else
			{
				data_.reset( );
			}
		}
		return *this;
	}

	nd_array& operator=( nd_array&& other ) noexcept = default;

	template<typename... Indices>
	reference operator( )( Indices... indices )
	{
		static_assert( sizeof...( indices ) <= MaxRank, "Too many indices" );
		return data_[compute_offset( indices... )];
	}

	template<typename... Indices>
	const_reference operator( )( Indices... indices ) const
	{
		static_assert( sizeof...( indices ) <= MaxRank, "Too many indices" );
		return data_[compute_offset( indices... )];
	}

	class nd_span
	{
	public:
		nd_span( pointer data, const std::array<size_type, MaxRank>& extents, const std::array<size_type, MaxRank>& strides, size_type rank )
		    : data_( data )
		    , extents_( extents )
		    , strides_( strides )
		    , rank_( rank )
		{
		}

		template<typename... Indices>
		reference operator( )( Indices... indices )
		{
			return data_[compute_offset( indices... )];
		}

		template<typename... Indices>
		const_reference operator( )( Indices... indices ) const
		{
			return data_[compute_offset( indices... )];
		}

		size_type extent( size_type dim ) const
		{
			if( dim >= rank_ )
			{
				throw std::out_of_range( "Dimension out of range" );
			}
			return extents_[dim];
		}

		size_type rank( ) const { return rank_; }

		pointer data( ) { return data_; }
		const_pointer data( ) const { return data_; }

	private:
		pointer data_;
		std::array<size_type, MaxRank> extents_;
		std::array<size_type, MaxRank> strides_;
		size_type rank_;

		template<typename... Indices>
		size_type compute_offset( Indices... indices ) const
		{
			size_type idx[]  = { static_cast<size_type>( indices )... };
			size_type offset = 0;
			for( size_t i = 0; i < sizeof...( indices ); ++i )
			{
				if( idx[i] >= extents_[i] )
				{
					throw std::out_of_range( "Index out of bounds" );
				}
				offset += idx[i] * strides_[i];
			}
			return offset;
		}
	};

	nd_span subspan( std::initializer_list<std::pair<size_type, size_type>> ranges )
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

		return nd_span( data_.get( ) + offset, new_extents, new_strides, rank_ );
	}

	nd_span subspan( size_type dim, size_type start, size_type end )
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

		return nd_span( data_.get( ) + offset, new_extents, strides_, rank_ );
	}

	nd_span slice( size_type dim, size_type index )
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

		return nd_span( data_.get( ) + offset, new_extents, new_strides, new_rank );
	}

	size_type extent( size_type dim ) const
	{
		if( dim >= rank_ )
		{
			throw std::out_of_range( "Dimension out of range" );
		}
		return extents_[dim];
	}

	size_type size( ) const { return size_; }
	size_type rank( ) const { return rank_; }
	constexpr size_type max_rank( ) const { return MaxRank; }

	pointer data( ) { return data_.get( ); }
	const_pointer data( ) const { return data_.get( ); }

	void fill( const T& value ) { std::fill( data_.get( ), data_.get( ) + size_, value ); }

	template<typename Func>
	void apply( Func&& func )
	{
		for( size_t i = 0; i < size_; ++i )
		{
			data_[i] = func( data_[i] );
		}
	}

private:
	std::unique_ptr<T[]> data_;
	std::array<size_type, MaxRank> extents_;
	std::array<size_type, MaxRank> strides_;
	size_type size_;
	size_type rank_;

	void compute_strides( )
	{
		if( rank_ == 0 )
			return;

		strides_[rank_ - 1] = 1;
		for( size_t i = rank_ - 1; i > 0; --i )
		{
			strides_[i - 1] = strides_[i] * extents_[i];
		}

		for( size_t i = rank_; i < MaxRank; ++i )
		{
			strides_[i] = 0;
		}
	}

	size_type compute_size( ) const
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

	template<typename... Indices>
	size_type compute_offset( Indices... indices ) const
	{
		size_type idx[]  = { static_cast<size_type>( indices )... };
		size_type offset = 0;
		for( size_t i = 0; i < sizeof...( indices ); ++i )
		{
			if( idx[i] >= extents_[i] )
			{
				throw std::out_of_range( "Index out of bounds" );
			}
			offset += idx[i] * strides_[i];
		}
		return offset;
	}
};
