#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "Common.h"
#include "retinaDetector.h"
#include "sharedBuffer.h"
#include "postDetections.h"
#include "caffePlugin.h"
#include <algorithm>

CFactory<CRetinaEngine> g_retinaCreator("retina");


std::vector<int> get_dim_size(Dims dim)
{
	std::vector<int> size;
	for (int i = 0; i < dim.nbDims; ++i)
		size.emplace_back(dim.d[i]);
	return size;
}

int total_size(std::vector<int> dim)
{
	int size = 1 * sizeof(float);
	for (auto d : dim)
		size *= d;
	return size;
}


CRetinaEngine::CRetinaEngine()
{
	std::string detectionModelPrefix = m_pconfiger->readValue("detectionModelPrefix");
	_setBatchSize(m_pconfiger->readValue<int>("detectionBatchSize"));
	_loadNetwork(m_modelPath + detectionModelPrefix, getCudaStream());

	auto engine = getCudaEngine();
	for (int b = 0; b < engine->getNbBindings(); ++b)
	{
		if (!engine->bindingIsInput(b))
		{
			LayerInfo l;
			l.name = engine->getBindingName(b);
			Dims dim_output = engine->getBindingDimensions(b);
			l.dim = get_dim_size(dim_output);
			l.size = total_size(l.dim);
			l.index = b;
			output_layer.emplace_back(l);
		}
	}
	cudaMallocHost((void**)&m_pcls, output_layer[output_layer.size() - 2].size);
	cudaMallocHost((void**)&m_pbox, output_layer.back().size);
	cudaMallocHost((void**)&m_pclsVice, output_layer[output_layer.size() - 2].size);
	cudaMallocHost((void**)&m_pboxVice, output_layer.back().size);
	float data0[8] = { -248,-248,263,263, -120,-120,135,135 };
	float data1[8] = { -56,-56,71,71, -24,-24,39,39 };
	float data2[8] = { -8,-8,23,23, 0,0,15,15 };
	m_anchors[0].Init(32, 2, data0);
	m_anchors[1].Init(16, 2, data1);
	m_anchors[2].Init(8, 2, data2);
}

CRetinaEngine::~CRetinaEngine()
{
	cudaFreeHost(m_pcls);
	cudaFreeHost(m_pbox);
	cudaFreeHost(m_pclsVice);
	cudaFreeHost(m_pboxVice);
}


void nms_cpu(std::vector<Anchor>& boxes, float threshold, std::vector<Anchor>& voboxes) {
	voboxes.resize(0);
	if (boxes.size() == 0)
		return;
	std::vector<size_t> idx(boxes.size());

	for (unsigned i = 0; i < idx.size(); i++)
	{
		idx[i] = i;
	}

	//descending sort
	std::sort(boxes.begin(), boxes.end(), std::greater<Anchor>());

	while (idx.size() > 0)
	{
		int good_idx = idx[0];
		voboxes.push_back(boxes[good_idx]);

		std::vector<size_t> tmp = idx;
		idx.clear();
		for (unsigned i = 1; i < tmp.size(); i++)
		{
			int tmp_i = tmp[i];
			float inter_x1 = std::max(boxes[good_idx][0], boxes[tmp_i][0]);
			float inter_y1 = std::max(boxes[good_idx][1], boxes[tmp_i][1]);
			float inter_x2 = std::min(boxes[good_idx][2], boxes[tmp_i][2]);
			float inter_y2 = std::min(boxes[good_idx][3], boxes[tmp_i][3]);

			float w = std::max((inter_x2 - inter_x1 + 1), 0.0F);
			float h = std::max((inter_y2 - inter_y1 + 1), 0.0F);

			float inter_area = w * h;
			float area_1 = (boxes[good_idx][2] - boxes[good_idx][0] + 1) * (boxes[good_idx][3] - boxes[good_idx][1] + 1);
			float area_2 = (boxes[tmp_i][2] - boxes[tmp_i][0] + 1) * (boxes[tmp_i][3] - boxes[tmp_i][1] + 1);
			float o = inter_area / (area_1 + area_2 - inter_area);
			if (o <= threshold)
				idx.push_back(tmp_i);
		}
	}
}


void converDetections(float* vobox, std::vector<Anchor>& vdetections, int vclassId)
{
	for (const auto& d : vdetections)
	{
		if (d.score < 0.01) continue;
		*++vobox = vclassId;
		*++vobox = d.score;
		memcpy(++vobox, &d.finalbox, 4 * sizeof(float)); vobox += 4;
	}
}

void adjustFace(std::vector<Anchor>& vfaces)
{//bbox_adapt_R.d
	cv::Matx44f bbox_adapt_R(0.355067475559597, 0.119262390148535, -0.355067475560677, -0.119262390148450,
		0.332986547478306, 0.246449822265788, -0.332986547479770, -0.246449822265689,
		-0.331071700057979, -0.146350441272887, 0.331071700059095, 0.146350441272787,
		-0.145853845175441, -0.399418765180059, 0.145853845176905, 0.399418765179917);
	for (auto& f : vfaces)
	{
		if (f.score < 0.01)continue;
		auto& b = f.finalbox;
		cv::Vec2f center((b.x + b.width) / 2, (b.y + b.height) / 2);
		float length = 4 / (b.width - b.x + b.height - b.y);
		cv::Vec4f boxNorm(b.x - center[0], b.y - center[1], b.width - center[0], b.height - center[1]);
		boxNorm *= length;
		boxNorm = bbox_adapt_R*boxNorm;
		boxNorm *= 1 / length;
		center[0] += (boxNorm[0] + boxNorm[2]) *0.5;
		center[1] += (boxNorm[1] + boxNorm[3]) *0.5;
		float width = boxNorm[2] - boxNorm[0];
		float heght = boxNorm[3] - boxNorm[1];
		int len = sqrt(width*width + heght*heght)*0.85 + 0.5;
		float x = center[0] - len*0.5;
		float y = center[1] - len*0.43;
		if (x < 0) x = 0;
		if (y < 0)y = 0;
		b.width = x + len;
		b.height = y + len;
	}
}

const int g_width = 960;
const int g_heght = 540;
int reFilter(std::vector<Anchor>& viodetections, float vHeight, float vWidth)
{
	static SFilterPara para{0.1,100,2};
	if (para.area < 101)
	{
		CConfiger* pconf = CConfiger::getOrCreateConfiger();
		std::vector<float> scoreAreaAspect;
		splitStringAndConvert(pconf->readValue("scoreAreaAspect"), scoreAreaAspect, ',');
		if (scoreAreaAspect.size()<3)
			std::cout << "error: invalid scoreAreaAspect in detector configurations" << std::endl;
		else
		{
			para.confidence = scoreAreaAspect[0];
			para.area = scoreAreaAspect[1];
			para.aspect = scoreAreaAspect[2];
		}
	}
	vWidth /= g_width;
	vHeight /= g_heght;
	int numRemain = 0;
	for (auto& d : viodetections)
	{
		if (d.score < para.confidence) { d.score = 0; continue; }
		auto& box = d.finalbox;
		box.x *= vWidth; box.width *= vWidth;
		box.y *= vHeight; box.height *= vHeight;

		float width = box.width - box.x;
		float heght = box.height - box.y;
		if (width*heght < para.area) { d.score = 0; continue; }
		float aspect = width / heght;
		if (aspect < 1) aspect = 1 / aspect;
		if (para.aspect < aspect) { d.score = 0; continue; }
// 		std::cout << d.score << ": "<< box.x << ", " << box.y << ", " << box.width << ", " << box.height << std::endl;
		numRemain++;
	}
	return numRemain;
}


void CRetinaEngine::postProcessV(std::vector<CDataShared*>& vmodelInputes, CDataShared* voutput)
{
	for (int i = 0; i < vmodelInputes.size(); ++i)
	{
		CImage* pimage = (CImage*)vmodelInputes[i];
		int L = pimage->modifiedLength;
		__getBoxes(resultBufferHost + i*modelOutputBytes / sizeof(float), L<0?-L:pimage->height, L>0?L:pimage->width, i);
	}
	CImageDetections* pout = (CImageDetections*)voutput;
	for (int i = 0; i < vmodelInputes.size(); ++i, pout++)
		getheadFacePair(resultBufferHost + i*modelOutputBytes / sizeof(float), pout);
}

void CRetinaEngine::__getBoxes(float* vobox, int vHeight, int vWidth, int vBatchIndex)
{
	proposals[0].resize(0);
	proposals[1].resize(0);
	auto& buffers = modelIObuffers;
// 	for (int i = 0; i < output_layer.size(); ++i)
// 	{
// 		auto& o = output_layer[i];
// 		CCaffePlugin::_printGPUdata((float*)buffers[o.index], o.size / 4, getCudaStream(), std::to_string(i));
// 	}	exit(0);

	__download(0, m_pclsVice, m_pboxVice, vBatchIndex);
	for (int i = 0; i < output_layer.size() / 2; ++i)
	{
		auto& lbox = output_layer[i * 2 + 1];
		cudaStreamSynchronize(getCudaStream());
		std::swap(m_pclsVice, m_pcls); std::swap(m_pboxVice, m_pbox);
		if (i < output_layer.size() / 2 - 1) __download(i * 2 + 2, m_pclsVice, m_pboxVice, vBatchIndex);
		m_anchors[i / 2].FilterAnchor(m_pcls, m_pbox, lbox.dim[2], lbox.dim[1], lbox.dim[0], proposals[i % 2]);
	}
	nms_cpu(proposals[0], m_nms_threshold, facesHeads[0]);
	nms_cpu(proposals[1], m_nms_threshold, facesHeads[1]);

	int numFace = reFilter(facesHeads[0], vHeight, vWidth);
	int numHead = reFilter(facesHeads[1], vHeight, vWidth);
	adjustFace(facesHeads[0]);
	converDetections(vobox, facesHeads[0], 2);
	converDetections(vobox + 7 * numFace, facesHeads[1], 1);
// 	std::cout << "faces: " << numFace << ", heads" << numHead << std::endl;
	*vobox = numFace + numHead;
}

void CRetinaEngine::__download(int vindex, float* vcls, float* vbox, int vBatchIndex/*=0*/)
{
	auto& vbuffers = modelIObuffers;
	auto& lcls = output_layer[vindex];
	auto& lbox = output_layer[vindex + 1];
	cudaMemcpyAsync(vcls, (char*)vbuffers[lcls.index]+vBatchIndex*lcls.size, lcls.size, cudaMemcpyDeviceToHost, getCudaStream());
	cudaMemcpyAsync(vbox, (char*)vbuffers[lbox.index]+vBatchIndex*lbox.size, lbox.size, cudaMemcpyDeviceToHost, getCudaStream());
}