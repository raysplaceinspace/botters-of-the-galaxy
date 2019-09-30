#pragma once
#include <assert.h>

template<typename value_t>
class Array {
	const value_t* _data;
	int _length;

public:
	__device__
	Array(const value_t* data, int length):
		_data(data),
		_length(length) {
	}

	__device__ int length() const { return this->_length; }

	__device__
	value_t& at(int i) { return const_cast<value_t&>(this->_data[i]); }

	__device__
	const value_t& at(int i) const { return this->_data[i]; }
};

template<typename value_t>
class Array2D {
	const value_t* _data;
	int _length0;
	int _length1;

public:
	__device__
	Array2D(const value_t* data, int length0, int length1):
		_data(data),
		_length0(length0),
		_length1(length1) {
	}

	__device__ int length0() const { return this->_length0; }
	__device__ int length1() const { return this->_length1; }

	__device__
	int index(int i, int j) const {
		/*
		length0 = 2 (num axes)
		length1 = 10 (num data points)

		[1, 4] -> 14
		[1, 5] -> 15

		*/
		assert(i <= this->_length0 && j <= this->_length1);
		return i * this->_length1 + j;
	}

	__device__
	Array<value_t> slice(int i) {
		return Array<value_t>(&this->_data[i * this->_length1], this->_length1);
	}

	__device__
	const Array<value_t> slice(int i) const {
		return Array<value_t>(&this->_data[i * this->_length1], this->_length1);
	}

	__device__
	value_t& at(int i, int j) {
		return const_cast<value_t&>(this->_data[this->index(i, j)]);
	}

	__device__
	const value_t& at(int i, int j) const {
		return this->_data[this->index(i, j)];
	}
};

template<typename value_t>
class Array3D {
	const value_t* _data;
	int _length0;
	int _length1;
	int _length2;

public:
	__device__
	Array3D(const value_t* data, int length0, int length1, int length2):
		_data(data),
		_length0(length0),
		_length1(length1),
		_length2(length2) {
	}

	__device__ int length0() const { return this->_length0; }
	__device__ int length1() const { return this->_length1; }
	__device__ int length2() const { return this->_length2; }

	__device__ int index(int i, int j, int k) const {
		/*
		length0 = 2 (num axes)
		length1 = 100 (num data points)
		length2 = 5 (num classes)

		[1, 10, 3] -> 2 * 100 * 5 + 10 * 5 + 3
		*/

		assert(i <= this->_length0 && j <= this->_length1 && k <= this->_length2);
		return i * this->_length1 * this->_length2 + j * this->_length2 + k;
	}

	__device__
	value_t& at(int i, int j, int k) {
		return const_cast<value_t&>(this->_data[this->index(i, j, k)]);
	}

	__device__
	const value_t& at(int i, int j, int k) const {
		return this->_data[this->index(i, j, k)];
	}
};

template<typename value_t>
class Array4D {
	const value_t* _data;
	int _length0;
	int _length1;
	int _length2;
	int _length3;

public:
	__device__
	Array4D(const value_t* data, int length0, int length1, int length2, int length3):
		_data(data),
		_length0(length0),
		_length1(length1),
		_length2(length2),
		_length3(length3) {
	}

	__device__ int length0() const { return this->_length0; }
	__device__ int length1() const { return this->_length1; }
	__device__ int length2() const { return this->_length2; }
	__device__ int length3() const { return this->_length3; }

	__device__
	int index(int i, int j, int k, int l) const {
		/*
		length0 = 10 (num open nodes)
		length1 = 2 (num axes)
		length2 = 32 (num categories)
		length3 = 5 (num classes)

		[3, 1, 7, 6] ->
			3 * 5 * 32 * 2 +
			1 * 5 * 32 +
			7 * 5 +
			6
		*/
		assert(i <= this->_length0 && j <= this->_length1 && k <= this->_length2 && l <= this->_length3);
		return
			i * this->_length3 * this->_length2 * this->_length1 +
			j * this->_length3 * this->_length2 +
			k * this->_length3 +
			l;
	}

	__device__
	value_t& at(int i, int j, int k, int l) {
		return const_cast<value_t&>(this->_data[this->index(i, j, k, l)]);
	}

	__device__
	const value_t& at(int i, int j, int k, int l) const {
		return this->_data[this->index(i, j, k, l)];
	}
};