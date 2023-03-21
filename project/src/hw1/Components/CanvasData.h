#pragma once

#include <UGM/UGM.h>

struct CanvasData {
	std::vector<Ubpa::pointf2> points;
	Ubpa::valf2 scrolling{ 0.f,0.f };
	bool opt_enable_grid{ true };
	bool opt_enable_context_menu{ true };
	bool adding_line{ false };


	bool opt_enable_adding_points{ true };

	bool opt_polynomial_fitting{ false };
	bool opt_gaussian_fitting{ false };
	bool opt_polynomial_regression{ false };
	bool opt_ridge_regression{ false };

	float sigma{ 5.0f };
	int span{ 0 };
	float lambda{ 1.0f };

	std::vector<Ubpa::pointf2> samplePoints;

	//bool isMenuItemOpen{ false };
};

#define MY_RED IM_COL32(255, 0, 0, 255)
#define MY_YELLOW IM_COL32(255, 255, 0, 255)
#define MY_CYAN IM_COL32(0, 255, 255, 255)
#define MY_GREEN IM_COL32(0, 255, 0, 255)
#define MY_WHITE IM_COL32(244, 244, 244, 255)

#define STEP 0.1f
#define MIN_CANVAS_SIZE 80.0f

#include "details/CanvasData_AutoRefl.inl"
