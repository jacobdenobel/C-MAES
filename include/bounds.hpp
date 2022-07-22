#pragma once

#include "common.hpp"
#include "sampling.hpp"

namespace bounds {
	
	struct BoundCorrection {
		Vector lb, ub, db;
		double diameter;
		size_t n_out_of_bounds = 0;

		BoundCorrection(const size_t dim) : 
			lb(Vector::Ones(dim) * -5), ub(Vector::Ones(dim) * 5), db(ub - lb),
			diameter((ub - lb).norm()) {}

		virtual void correct(Matrix& X, Matrix& Y, const Vector& s, const Vector& m)  = 0;

	};

	struct NoCorrection : BoundCorrection {
		using BoundCorrection::BoundCorrection;

		void correct(Matrix& X, Matrix& Y, const Vector& s, const Vector& m)  override {}
	};

	struct CountOutOfBounds : BoundCorrection {
		using BoundCorrection::BoundCorrection;

		void correct(Matrix& X, Matrix& Y, const Vector& s, const Vector& m)  override {
			n_out_of_bounds = 0;
			for (auto i = 0; i < X.cols(); ++i) {
				const auto oob = X.col(i).array() < lb.array() || X.col(i).array() > ub.array();
				n_out_of_bounds += oob.any();				
			}
		}
	};

	inline double modulo2(const int x) {
		return static_cast<double>(x % 2);
	};

	struct COTN : BoundCorrection {
		sampling::Gaussian sampler;

		COTN(const size_t dim): BoundCorrection(dim), sampler(dim, std::normal_distribution<double>(0, 1.0 / 3)) {}

		void correct(Matrix& X, Matrix& Y, const Vector& s, const Vector& m) override {
			n_out_of_bounds = 0;

			for (auto i = 0; i < X.cols(); ++i) {
				const auto oob = X.col(i).array() < lb.array() || X.col(i).array() > ub.array();
				n_out_of_bounds += oob.any();
				if (oob.any()) {
					X.col(i) = (oob).select((X.col(i) - lb).cwiseQuotient(db), X.col(i));
					X.col(i) = (oob).select(
						lb.array() + db.array() * ((X.col(i).array() > 0).cast<double>() - sampler().array()).abs(), X.col(i)
					);
					Y.col(i) = (X.col(i) - m) / s(i);
				}				
			}
		}
	};

	struct Mirror : BoundCorrection {
		using BoundCorrection::BoundCorrection;

		void correct(Matrix& X, Matrix& Y, const Vector& s, const Vector& m)  override {
			n_out_of_bounds = 0;

			for (auto i = 0; i < X.cols(); ++i) {
				const auto oob = X.col(i).array() < lb.array() || X.col(i).array() > ub.array();
				n_out_of_bounds += oob.any();
				if (oob.any()) {
					X.col(i) = (oob).select((X.col(i) - lb).cwiseQuotient(db), X.col(i));
					X.col(i) = (oob).select(
						lb.array() + (db.array() * (X.col(i).array() - X.col(i).array().floor() - X.col(i).array().unaryExpr(&modulo2)).abs()), X.col(i)
					);
					Y.col(i) = (X.col(i) - m) / s(i);
				}
			}
		}
	};

	struct UniformResample : BoundCorrection {
		sampling::Random<std::uniform_real_distribution<>> sampler;

		UniformResample(const size_t dim): BoundCorrection(dim), sampler(dim) {}

		void correct(Matrix& X, Matrix& Y, const Vector& s, const Vector& m)  override {
			n_out_of_bounds = 0;

			for (auto i = 0; i < X.cols(); ++i) {
				const auto oob = X.col(i).array() < lb.array() || X.col(i).array() > ub.array();
				n_out_of_bounds += oob.any();
				if (oob.any()) {
					X.col(i) = (oob).select(lb + sampler().cwiseProduct(db), X.col(i));
					Y.col(i) = (X.col(i) - m) / s(i);
				}
			}
		}
	};

	struct Saturate : BoundCorrection {
		using BoundCorrection::BoundCorrection;

		void correct(Matrix& X, Matrix& Y, const Vector& s, const Vector& m)  override {
			n_out_of_bounds = 0;

			for (auto i = 0; i < X.cols(); ++i) {
				const auto oob = X.col(i).array() < lb.array() || X.col(i).array() > ub.array();
				n_out_of_bounds += oob.any();
				if (oob.any()) {
					X.col(i) = (oob).select((X.col(i) - lb).cwiseQuotient(db), X.col(i));
					X.col(i) = (oob).select(
						lb.array() + db.array() * (X.col(i).array() > 0).cast<double>(), X.col(i)
					);
					Y.col(i) = (X.col(i) - m) / s(i);
				}
			}
		}
	};

	struct Toroidal : BoundCorrection {
		using BoundCorrection::BoundCorrection;

		void correct(Matrix& X, Matrix& Y, const Vector& s, const Vector& m)  override {
			n_out_of_bounds = 0;

			for (auto i = 0; i < X.cols(); ++i) {
				const auto oob = X.col(i).array() < lb.array() || X.col(i).array() > ub.array();
				n_out_of_bounds += oob.any();
				if (oob.any()) {
					X.col(i) = (oob).select((X.col(i) - lb).cwiseQuotient(db), X.col(i));
					X.col(i) = (oob).select(
						lb.array() + db.array() * (X.col(i).array() - X.col(i).array().floor()).abs(), X.col(i)
					);
					Y.col(i) = (X.col(i) - m) / s(i);
				}
			}
		}
	};


	enum class CorrectionMethod {
		NONE, COUNT, MIRROR, COTN, UNIFORM_RESAMPLE, SATURATE, TOROIDAL
	};

	inline std::shared_ptr<BoundCorrection> get(const size_t dim, const CorrectionMethod& m) {
		switch (m)
		{
		case CorrectionMethod::NONE:
			return std::make_shared<NoCorrection>(dim);
		case CorrectionMethod::COUNT:
			return std::make_shared<CountOutOfBounds>(dim);
		case CorrectionMethod::MIRROR:
			return std::make_shared<Mirror>(dim);
		case CorrectionMethod::COTN:
			return std::make_shared<COTN>(dim);
		case CorrectionMethod::UNIFORM_RESAMPLE:
			return std::make_shared<UniformResample>(dim);
		case CorrectionMethod::SATURATE:
			return std::make_shared<Saturate>(dim);
		case CorrectionMethod::TOROIDAL:
			return std::make_shared<Toroidal>(dim);
		}
	};
}


