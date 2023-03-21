#include "CanvasSystem.h"

#include "../Components/CanvasData.h"

#include <_deps/imgui/imgui.h>

#include <Eigen/Core>
#include <Eigen/Dense>
#include <cmath>
#include <tuple>


using namespace Ubpa;

inline std::tuple<float, float> getMinMax(std::vector<Ubpa::pointf2>& points) {
    float min = std::numeric_limits<float>::max();
    float max = std::numeric_limits<float>::min();
    for (auto& p : points) {
        if (p[0] < min) {
            min = p[0];
        }
        if (p[0] > max) {
            max = p[0];
        }
    }
    return std::make_tuple(min, max);
}

inline Eigen::RowVectorXf getPolynomialRow(float x, int size) {
    Eigen::RowVectorXf row(size);
    for (int k = 0; k < size; k++) {
        row[k] = std::pow(x, k);
    }
    return row;
}

inline Eigen::RowVectorXf getGaussianBaseRow(Eigen::VectorXf xSamples, float x, float sigma) {
    Eigen::ArrayXf arr = xSamples.array();
    Eigen::ArrayXf gaussian = (-(arr - x).square() / (2 * sigma * sigma)).exp();
    Eigen::RowVectorXf result(xSamples.size() + 1);
    result << 1.0f, gaussian.matrix().transpose();
    return result;
}

inline int getRenderingpointsX(std::vector<Ubpa::pointf2>& points, std::vector<float> & target) {
    auto [xMin, xMax] = getMinMax(points);
    int renderingSize = int((xMax - xMin) / STEP) + 1;
    //std::vector<float> xRenderingPoints(renderingSize);
    target.resize(renderingSize);

    target[0] = xMin;
    int ind = 1;
    while (ind < renderingSize) {
        target[ind] = target[ind - 1] + STEP;
        ind += 1;
    }
    return renderingSize;
}

inline void connectPointsWithPolyline(
    ImDrawList *dl, const ImVec2 &origin, const std::vector<float>& xPoints, const std::vector<float>& yPoints, ImU32 color
) {
    std::vector<ImVec2> canvasPoints(xPoints.size());
    for (int i = 0; i < canvasPoints.size(); i += 1) {
        canvasPoints[i] = ImVec2(origin.x + xPoints[i], origin.y - yPoints[i]);
    }
    dl->AddPolyline(canvasPoints.data(), canvasPoints.size(), color, false, 3.0f);
}

void polynomialFittingInterpolation(ImDrawList *dl, std::vector<Ubpa::pointf2>& points, const ImVec2& origin, ImU32 color = MY_RED) {
    int n = points.size();
    Eigen::MatrixXf X(n, n);
    Eigen::VectorXf y(n);

    int j = 0;
    for (auto& p : points) {
        y[j] = -p[1];
        X.row(j) = getPolynomialRow(p[0], n);
        j += 1;
    }
    
    Eigen::VectorXf polynomialCoefficients = X.householderQr().solve(y);

    std::vector<float> xRenderingPoints;
    int renderingSize = getRenderingpointsX(points, xRenderingPoints);

    std::vector<float> yRenderingPoints(renderingSize);
    for (int i = 0; i < renderingSize; i += 1) {
        yRenderingPoints[i] = getPolynomialRow(xRenderingPoints[i], n).dot(polynomialCoefficients);
    }
    
    connectPointsWithPolyline(dl, origin, xRenderingPoints, yRenderingPoints, color);
}

void gaussianFittingInterpolation(ImDrawList *dl, std::vector<Ubpa::pointf2>& points, const ImVec2& origin, float sigma, ImU32 color = MY_YELLOW) {
    int n = points.size();
    Eigen::VectorXf y(n + 1);
    Eigen::VectorXf x(n);
    Eigen::MatrixXf X(n + 1, n + 1);

    int ind = 0;
    for (auto& p : points) {
        y[ind] = -p[1];
        x[ind] = p[0];
        ind += 1;
    }
    y[ind] = (y[ind - 1] + y[ind - 2]) / 2;
    for (int i = 0; i < n; i += 1) {
        X.row(i) = getGaussianBaseRow(x, x[i], sigma);
    }
    X.row(n) = getGaussianBaseRow(x, (x[n - 1] + x[n - 2]) / 2, sigma);

    Eigen::VectorXf gaussianCoefficients = X.householderQr().solve(y);

    std::vector<float> xRenderingPoints;
    int renderingSize = getRenderingpointsX(points, xRenderingPoints);

    std::vector<float> yRenderingPoints(renderingSize);
    for (int i = 0; i < renderingSize; i += 1) {
        yRenderingPoints[i] = getGaussianBaseRow(x, xRenderingPoints[i], n).dot(gaussianCoefficients);
    }

    connectPointsWithPolyline(dl, origin, xRenderingPoints, yRenderingPoints, color);
}

void ridgeRegressionInterpolation(ImDrawList *dl, std::vector<Ubpa::pointf2>& points, const ImVec2&origin, int span, float lambda, ImU32 color = MY_GREEN) {
    int n = points.size();
    if (span > n) {
        span = n;
    }

    Eigen::MatrixXf X(n, span);
    Eigen::VectorXf y(n);
    int ind = 0;
    for (auto& p : points) {
        y[ind] = -p[1];
        X.row(ind) = getPolynomialRow(p[0], span);
        ind += 1;
    }

    Eigen::MatrixXf I = Eigen::MatrixXf::Identity(span, span);

    Eigen::VectorXf polynomialCoefficients = (X.transpose() * X + lambda * I).householderQr().solve(X.transpose() * y);

    std::vector<float> xRenderingPoints;
    int renderingSize = getRenderingpointsX(points, xRenderingPoints);

    std::vector<float> yRenderingPoints(renderingSize);
    for (int i = 0; i < renderingSize; i += 1) {
        yRenderingPoints[i] = getPolynomialRow(xRenderingPoints[i], span).dot(polynomialCoefficients);
    }

    connectPointsWithPolyline(dl, origin, xRenderingPoints, yRenderingPoints, color);
}

void polynomialRegressionInterpolation(ImDrawList* dl, std::vector<Ubpa::pointf2>& points, const ImVec2& origin, int span, ImU32 color = MY_CYAN) {
    ridgeRegressionInterpolation(dl, points, origin, span, 0.0f, color);
}

inline bool isMouseInCanvas(ImGuiIO& io, const ImVec2 &p0, const ImVec2 &p1) {
    float xBoundMin = p0.x;
    float xBoundMax = p1.x;
    float yBoundMin = p0.y;
    float yBoundMax = p1.y;
    float x = io.MousePos.x;
    float y = io.MousePos.y;
    return (x > xBoundMin) && (x < xBoundMax) && (y > yBoundMin) && (y < yBoundMax);
}

void CanvasSystem::OnUpdate(Ubpa::UECS::Schedule& schedule) {
    schedule.RegisterCommand([](Ubpa::UECS::World* w) {
        auto data = w->entityMngr.GetSingleton<CanvasData>();
        if (!data)
            return;

        if (ImGui::Begin("Canvas")) {
            if (ImGui::CollapsingHeader("Enable")) {
                ImGui::Checkbox("Enable grid", &data->opt_enable_grid);
                //ImGui::Checkbox("Enable context menu", &data->opt_enable_context_menu);
                ImGui::Checkbox("Enable adding points", &data->opt_enable_adding_points);
            }

            if (ImGui::CollapsingHeader("Interpolations")) {
                ImGui::PushStyleColor(ImGuiCol_Text, MY_RED);
                ImGui::Checkbox("Show Polynomial Fitting    ", &data->opt_polynomial_fitting);
                ImGui::PopStyleColor();

                ImGui::SameLine();
                ImGui::PushStyleColor(ImGuiCol_Text, MY_YELLOW);
                ImGui::Checkbox("Show Gaussian Fitting", &data->opt_gaussian_fitting);
                ImGui::PopStyleColor();

                ImGui::PushStyleColor(ImGuiCol_Text, MY_CYAN);
                ImGui::Checkbox("Show Polynomial Regression ", &data->opt_polynomial_regression);
                ImGui::PopStyleColor();

                ImGui::SameLine();
                ImGui::PushStyleColor(ImGuiCol_Text, MY_GREEN);
                ImGui::Checkbox("Show Ridge Regression", &data->opt_ridge_regression);
                ImGui::PopStyleColor();
            }

            if (ImGui::CollapsingHeader("Parameters")) {
                ImGui::SliderFloat("sigma", &data->sigma, 1.0f, 50.0f);
                ImGui::SliderInt("span", &data->span, 0, 10);
                ImGui::SliderFloat("lambda", &data->lambda, 0.0f, 1000.0f);
            }

            ImGui::Text("Mouse Left: click to add sample points");
            
            if (ImGui::Button("Remove One Sample Point")) {
                if (data->samplePoints.size() > 0) {
                    data->samplePoints.pop_back();
                }
            }
            ImGui::SameLine();
            if (ImGui::Button("Remove All Sample Points")) {
                if (data->samplePoints.size() > 0) {
                    data->samplePoints.clear();
                }
            }

            // Using InvisibleButton() as a convenience 1) it will advance the layout cursor and 2) allows us to use IsItemHovered()/IsItemActive()
            ImVec2 canvas_p0 = ImGui::GetCursorScreenPos();      // ImDrawList API uses screen coordinates!
            ImVec2 canvas_sz = ImGui::GetContentRegionAvail();   // Resize canvas to what's available
            if (canvas_sz.x < MIN_CANVAS_SIZE) canvas_sz.x = MIN_CANVAS_SIZE;
            if (canvas_sz.y < MIN_CANVAS_SIZE) canvas_sz.y = MIN_CANVAS_SIZE;
            ImVec2 canvas_p1 = ImVec2(canvas_p0.x + canvas_sz.x, canvas_p0.y + canvas_sz.y);

            // Draw border and background color
            ImGuiIO& io = ImGui::GetIO();
            ImDrawList* draw_list = ImGui::GetWindowDrawList();
            draw_list->AddRectFilled(canvas_p0, canvas_p1, IM_COL32(50, 50, 50, 255));
            draw_list->AddRect(canvas_p0, canvas_p1, IM_COL32(255, 255, 255, 255));

            // This will catch our interactions
            ImGui::InvisibleButton("canvas", canvas_sz, ImGuiButtonFlags_MouseButtonLeft | ImGuiButtonFlags_MouseButtonRight);
            const bool is_hovered = ImGui::IsItemHovered(); // Hovered
            const bool is_active = ImGui::IsItemActive();   // Held
            const ImVec2 origin(canvas_p0.x + data->scrolling[0], canvas_p0.y + data->scrolling[1]); // Lock scrolled origin
            const pointf2 mouse_pos_in_canvas(io.MousePos.x - origin.x, io.MousePos.y - origin.y);


            // Pan (we use a zero mouse threshold when there's no context menu)
            // You may decide to make that threshold dynamic based on whether the mouse is hovering something etc.
            const float mouse_threshold_for_pan = data->opt_enable_context_menu ? -1.0f : 0.0f;
            if (is_active && ImGui::IsMouseDragging(ImGuiMouseButton_Right, mouse_threshold_for_pan))
            {
                data->scrolling[0] += io.MouseDelta.x;
                data->scrolling[1] += io.MouseDelta.y;
            }


            // push the sample points
            ImVec2 drag_delta_left = ImGui::GetMouseDragDelta(ImGuiMouseButton_Left);
            float epsilon = 0.00001f;
            if (
                data->opt_enable_adding_points && ImGui::IsMouseReleased(ImGuiMouseButton_Left)
                && drag_delta_left.x < epsilon && drag_delta_left.y < epsilon
            ) {
                if (isMouseInCanvas(io, canvas_p0, canvas_p1)) {
                    data->samplePoints.push_back(mouse_pos_in_canvas);
                }
            }
            
            // Draw grid + all lines in the canvas
            draw_list->PushClipRect(canvas_p0, canvas_p1, true);
            if (data->opt_enable_grid)
            {
                const float GRID_STEP = 64.0f;
                for (float x = fmodf(data->scrolling[0], GRID_STEP); x < canvas_sz.x; x += GRID_STEP)
                    draw_list->AddLine(ImVec2(canvas_p0.x + x, canvas_p0.y), ImVec2(canvas_p0.x + x, canvas_p1.y), IM_COL32(200, 200, 200, 40));
                for (float y = fmodf(data->scrolling[1], GRID_STEP); y < canvas_sz.y; y += GRID_STEP)
                    draw_list->AddLine(ImVec2(canvas_p0.x, canvas_p0.y + y), ImVec2(canvas_p1.x, canvas_p0.y + y), IM_COL32(200, 200, 200, 40));
            }
            for (int n = 0; n < data->points.size(); n += 2)
                draw_list->AddLine(ImVec2(origin.x + data->points[n][0], origin.y + data->points[n][1]), ImVec2(origin.x + data->points[n + 1][0], origin.y + data->points[n + 1][1]), IM_COL32(255, 255, 0, 255), 2.0f);
            draw_list->PopClipRect();


            // do interpolation
            if (data->samplePoints.size() > 1) {
                if (data->opt_polynomial_fitting) {
                    polynomialFittingInterpolation(draw_list, data->samplePoints, origin);
                }
                if (data->opt_gaussian_fitting) {
                    gaussianFittingInterpolation(draw_list, data->samplePoints, origin, data->sigma);
                }
                if (data->opt_polynomial_regression) {
                    polynomialRegressionInterpolation(draw_list, data->samplePoints, origin, data->span + 1);
                }
                if (data->opt_ridge_regression) {
                    ridgeRegressionInterpolation(draw_list, data->samplePoints, origin, data->span + 1, data->lambda);
                }
            }

            // draw sample points
            for (auto& p : data->samplePoints) {
                draw_list->AddCircle(ImVec2(origin.x + p[0], origin.y + p[1]), 4.0f, MY_WHITE, 0, 2.5f);
            }

            ImGui::Text("mouse position in canvas coordinate0: %f", mouse_pos_in_canvas[0]);
            ImGui::Text("mouse position in canvas coordinate1: %f", mouse_pos_in_canvas[1]);
            ImGui::Text("#sample points: %d", data->samplePoints.size());
        }

        ImGui::End();
    });
}
