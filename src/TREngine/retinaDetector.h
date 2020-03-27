#pragma once
#include "modelEngine.h"
struct LayerInfo
{
	std::vector<int> dim;
	std::string name;
	int index;
	int size;
};
class CRect2f {
public:
	CRect2f(float x1, float y1, float x2, float y2) {
		val[0] = x1;
		val[1] = y1;
		val[2] = x2;
		val[3] = y2;
	}

	float& operator[](int i) {
		return val[i];
	}

	float operator[](int i) const {
		return val[i];
	}

	float val[4];

	void print() {
		printf("rect %f %f %f %f\n", val[0], val[1], val[2], val[3]);
	}
};
class Anchor {
public:
	bool operator<(const Anchor &t) const {
		return score < t.score;
	}

	bool operator>(const Anchor &t) const {
		return score > t.score;
	}

	float& operator[](int i) {
		assert(0 <= i && i <= 4);

		if (i == 0)
			return finalbox.x;
		if (i == 1)
			return finalbox.y;
		if (i == 2)
			return finalbox.width;
		if (i == 3)
			return finalbox.height;
	}

	float operator[](int i) const {
		assert(0 <= i && i <= 4);

		if (i == 0)
			return finalbox.x;
		if (i == 1)
			return finalbox.y;
		if (i == 2)
			return finalbox.width;
		if (i == 3)
			return finalbox.height;
	}

	cv::Rect_< float > anchor; // x1,y1,x2,y2
	float reg[4]; // offset reg
	cv::Point center; // anchor feat center
	float score; // cls score
	std::vector<cv::Point2f> pts; // pred pts

	cv::Rect_< float > finalbox; // final box res
};

class AnchorGenerator {
public:
	void Init(int stride, int vanchorNum, float* data)
	{
		m_anchorStride = stride; // anchor tile stride
		m_presetAnchors.push_back(CRect2f(data[0], data[1], data[2], data[3]));
		m_presetAnchors.push_back(CRect2f(data[4], data[5], data[6], data[7]));
		m_anchorNum = vanchorNum; // anchor type num
	}
	// filter anchors and return valid anchors
	int FilterAnchor(float* cls, float* vpbox, int w, int h, int c, std::vector<Anchor>& result)
	{
		for (int i = 0; i < h; ++i) {
			for (int j = 0; j < w; ++j) {
				int id = i * w + j;
				for (int a = 0; a < m_anchorNum; ++a)
				{
					float score = cls[(m_anchorNum + a)*w*h + id];
					if (score >= m_clsThreshold)
					{
						CRect2f boxAnchor(j * m_anchorStride + m_presetAnchors[a][0],
							i * m_anchorStride + m_presetAnchors[a][1],
							j * m_anchorStride + m_presetAnchors[a][2],
							i * m_anchorStride + m_presetAnchors[a][3]);
						//printf("%f %f %f %f\n", box[0], box[1], box[2], box[3]);
						CRect2f delta(vpbox[(a * 4 + 0)*w*h + id],
							vpbox[(a * 4 + 1)*w*h + id],
							vpbox[(a * 4 + 2)*w*h + id],
							vpbox[(a * 4 + 3)*w*h + id]);

						Anchor res;
						res.anchor = cv::Rect_<float>(boxAnchor[0], boxAnchor[1], boxAnchor[2], boxAnchor[3]);
						bbox_pred(boxAnchor, delta, res.finalbox);
						//printf("bbox pred\n");
						res.score = score;
						res.center = cv::Point(j, i);

						//printf("center %d %d\n", j, i);
						result.push_back(res);
					}
				}
			}
		}
		return 0;
	}

private:
	void bbox_pred(const CRect2f& anchor, const CRect2f& delta, cv::Rect_< float >& vobox)
	{
		float w = anchor[2] - anchor[0] + 1;
		float h = anchor[3] - anchor[1] + 1;
		float x_ctr = anchor[0] + 0.5 * (w - 1);
		float y_ctr = anchor[1] + 0.5 * (h - 1);

		float dx = delta[0];
		float dy = delta[1];
		float dw = delta[2];
		float dh = delta[3];

		float pred_ctr_x = dx * w + x_ctr;
		float pred_ctr_y = dy * h + y_ctr;
		float pred_w = std::exp(dw) * w;
		float pred_h = std::exp(dh) * h;

		vobox = cv::Rect_< float >(pred_ctr_x - 0.5 * (pred_w - 1.0),
			pred_ctr_y - 0.5 * (pred_h - 1.0),
			pred_ctr_x + 0.5 * (pred_w - 1.0),
			pred_ctr_y + 0.5 * (pred_h - 1.0));
	}

	int m_anchorStride; // anchor tile stride
	std::vector<CRect2f> m_presetAnchors;
	int m_anchorNum; // anchor type num
	float m_clsThreshold = 0.15;
};

class CSharedBuffer;
class CRetinaEngine : public CModelEngine
{
public:
	CRetinaEngine();
	~CRetinaEngine();

	virtual void postProcessV(std::vector<CDataShared*>& vmodelInputes, CDataShared* voutput) override;

private:
	float m_nms_threshold = 0.4;
	float *m_pcls, *m_pbox, *m_pclsVice, *m_pboxVice;
	std::vector<Anchor> proposals[2], facesHeads[2];
	void __getBoxes(float* vobox, int vHeight, int vWidth, int vBatchIndex);
	void __download(int vindex, float* vcls, float* vbox, int vBatchIndex=0);

	std::vector<float> m_buffer;
	AnchorGenerator m_anchors[3];
	std::vector<LayerInfo> output_layer;
};