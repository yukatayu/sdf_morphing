#include <vector>
#include <ranges>
#include <limits>
#include <queue>
#include <opencv2/opencv.hpp>

using SDF = std::vector<std::vector<float>>;

// メモ化再帰計算用の構造体
struct SDF_intermediate {
	// outside -> [0], inside -> [1]
	std::array<float, 2> min_distance = { std::numeric_limits<float>::infinity(), std::numeric_limits<float>::infinity() };
	std::array<std::optional<std::pair<int, int>>, 2> nearest_position;
	float getDistance() const noexcept {
		if (min_distance[1] > 0 && std::isfinite(min_distance[1]))
			return min_distance[1];
		return -min_distance[0];
	}
};

// SDFを画像としてデバッグ表示する
cv::Mat asMat(const SDF& sdf) {
	cv::Mat converted(sdf.size(), sdf.at(0).size(), CV_8U);
	for (auto&& [y, row] : sdf | std::views::enumerate)
		for (auto&& [x, element] : row | std::views::enumerate)
			converted.at<unsigned char>(y,x) = element + 128;
	return converted;
}

// SDFから画像を復元する
cv::Mat toImage(const SDF& sdf) {
	cv::Mat converted(sdf.size(), sdf.at(0).size(), CV_8U);
	for (auto&& [y, row] : sdf | std::views::enumerate)
		for (auto&& [x, element] : row | std::views::enumerate)
			converted.at<unsigned char>(y, x) = (element <= 0 ? 0 : 255);
	return converted;
}

// 2点間の距離を求める
float distance(std::pair<int, int> a, std::pair<int, int>b) {
	const int y_diff = (a.first - b.first);
	const int x_diff = (a.second - b.second);
	return std::sqrt(y_diff * y_diff + x_diff * x_diff);
}

// 2つのSDFを線形補間する
SDF lerp(const SDF& sdf0, const SDF& sdf1, float alpha) {
	return
		std::views::zip(sdf0, sdf1)
		| std::views::transform([alpha](const auto& row_zip) -> std::vector<float> {
				auto&& [row0, row1] = row_zip;
				return
					std::views::zip(row0, row1)
					| std::views::transform([alpha](const auto& element_zip) -> float {
							const auto& [element0, element1] = element_zip;
							return element0 * alpha + element1 * (1-alpha);
						})
					| std::ranges::to<std::vector<float>>();
			})
		| std::ranges::to<std::vector<std::vector<float>>>();
}

// 画像からSDFを計算する
SDF calculateSDF(cv::Mat input) {
	std::vector<std::vector<SDF_intermediate>> status(input.rows, std::vector<SDF_intermediate>(input.cols, {}));
	std::array<std::queue<std::pair<int, int>>, 2> bfs;
	// 初期化
	for (auto&& [y, row] : status | std::views::enumerate)
		for (auto&& [x, element] : row | std::views::enumerate) {
			const bool is_inside = input.at<unsigned char>(y, x) < 128;
			status[y][x].min_distance[is_inside] = 0;
			status[y][x].nearest_position[is_inside] = { y, x };
			bfs[is_inside].push({ y, x });
		}

	// イテレーション
	for (const int is_inside : {0, 1}) {
		while (!bfs[is_inside].empty()) {
			// 先頭要素を取り出す
			const auto [from_y, from_x] = bfs[is_inside].front();
			bfs[is_inside].pop();
			// 周囲のピクセルと比較
			for (const int dy : {-1, 0, 1})
				for (const int dx : {-1, 0, 1}) {
					// 自分自身とは比較しない
					if (dy == 0 && dx == 0)
						continue;
					// 伝播先の座標
					const int to_y = from_y + dy;
					const int to_x = from_x + dx;
					// 画像領域外アクセスを防ぐ
					if (to_y < 0 || input.rows <= to_y || to_x < 0 || input.cols <= to_x)
						continue;
					const auto& status_from = status.at(from_y).at(from_x);
					auto& status_to = status.at(to_y).at(to_x);
					// 距離を伝播
					if (status_from.nearest_position[is_inside]) {
						const float maybe_nearest_distance = distance({ to_y, to_x }, *status_from.nearest_position[is_inside]);
						if (maybe_nearest_distance < status_to.min_distance[is_inside]) {
							status_to.min_distance[is_inside] = maybe_nearest_distance;
							status_to.nearest_position[is_inside] = *status_from.nearest_position[is_inside];
							bfs[is_inside].push({ to_y, to_x });
						}
					}
				}
		}
	}

	// 集計
	return
		status
		| std::views::transform([](const std::vector<SDF_intermediate>& row) -> std::vector<float> {
			return
				row
				| std::views::transform([](const SDF_intermediate& element) -> float {
						return element.getDistance(); })
				| std::ranges::to<std::vector<float>>(); })
		| std::ranges::to<std::vector<std::vector<float>>>();
}

int main() {
	// 白黒画像として読み込み
	cv::Mat phase0 = cv::imread("phase0.png", 0);
	cv::Mat phase1 = cv::imread("phase1.png", 0);

	// SDF計算
	SDF sdf0 = calculateSDF(phase0);
	SDF sdf1 = calculateSDF(phase1);

	// ブレンドして連番出力
	constexpr int phase_count = 50;
	for (const int phase : std::views::iota(0, phase_count)) {
		float alpha = static_cast<float>(phase) / phase_count;
		SDF blended = lerp(sdf0, sdf1, alpha);
		cv::imwrite(std::format("blended_{:03}.png", phase), toImage(blended));
	}
}
