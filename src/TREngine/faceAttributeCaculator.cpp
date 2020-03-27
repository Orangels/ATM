#include "faceAttributeCaculator.h"
#include "Common.h"
#include "dataStructure.h"
#include "FileFunction.h"
CFactory<CFaceAttributeCalculator> g_faceAttributeCreator("CFaceAttributeCalculator");


CFaceAttributeCalculator::CFaceAttributeCalculator()
{
	std::string faceAttributeModelPrefix = m_pconfiger->readValue("faceAttributeModelPrefix");
	_setBatchSize(m_pconfiger->readValue<int>("faceAttributeBatchSize"));
	_loadNetwork(m_modelPath + faceAttributeModelPrefix, getCudaStream());
 	cudaMalloc((void**)&m_pointsBuffer, 219 * getBatchSize() * sizeof(float));
//	cudaMallocHost((void**)&m_pointsBuffer, 219 * getBatchSize() * sizeof(float));
	if (m_pconfiger->readValue("recognizationRunModel") != "nonuse")
	{
		m_pfaceRecognization = (CModelEngine*)CFactoryDirectory::getOrCreateInstance(true)->createProduct("CFaceRecognization");
		m_faceAttriOI.resize(2);
	}
}

CFaceAttributeCalculator::~CFaceAttributeCalculator()
{
}
#include "caffePlugin.h"
void CFaceAttributeCalculator::preProcessV(std::vector<CDataShared *>& vsrc)
{
	static int itest = 0; 
	std::vector<CDataShared *> batchFaces(vsrc.rbegin(), vsrc.rend());
	std::vector<CFaceKeyPoints *> facesOnSameImage;
	int offset = 0; 
	while (batchFaces.size())
	{
		CFace* pface = (CFace*)batchFaces.back()->m_pproducer;
		facesOnSameImage.resize(0);
		facesOnSameImage.emplace_back((CFaceKeyPoints *)batchFaces.back());
		batchFaces.pop_back();
		CImage* pimage = (CImage*)pface->m_pproducer;
		while (batchFaces.size())
		{
			pface = (CFace*)batchFaces.back()->m_pproducer;
			if (pface->m_pproducer == pimage)
			{
				facesOnSameImage.emplace_back((CFaceKeyPoints *)batchFaces.back());
				batchFaces.pop_back();
			}
			else break;
		}
		float* p68keypointsCPU = (float*)facesOnSameImage.front()->getClodData(true);
		bool isContinuousMemery = (facesOnSameImage.size()==1||facesOnSameImage.back()->getClodData(true) > facesOnSameImage.front()->getClodData(true));
		if (!isContinuousMemery)
		{
			p68keypointsCPU = m_pointsBuffer;
			for (int k = 0; k < facesOnSameImage.size(); ++k)
				cudaMemcpyAsync(p68keypointsCPU+219*k, facesOnSameImage[k]->getClodData(true), 219*sizeof(float), cudaMemcpyDeviceToDevice, getViceCudaStream());
		}
		
// 		for (int k = 0; k < facesOnSameImage.size(); ++k)
// 			outPutSum(p68keypointsCPU + k * 219, 219, "68points");

		affine(facesOnSameImage.size(), p68keypointsCPU, pimage, (float *)m_inputViceBuffer + offset*m_indims.c()*m_indims.h()*m_indims.w());
		offset += facesOnSameImage.size();
	}
}

#include "caffePlugin.h"
void CFaceAttributeCalculator::postProcessV(std::vector<CDataShared *>& vmodelInputes, CDataShared* voutput)
{
	int nIn = m_indims.c()*m_indims.h()*m_indims.w();
	int nOut = m_dimsOut.c()*m_dimsOut.h()*m_dimsOut.w();
	static int itest = 0;
// 	for (int k = 0; k < vmodelInputes.size(); ++k)
// 	{
// 		CCaffePlugin::_printGPUdata((float*)modelIObuffers.front() + k*nIn, nIn, getCudaStream(), std::to_string(itest)+ "faceInfor inputs");
// 		CCaffePlugin::_printGPUdata((float*)modelIObuffers.back() + k*nOut, nOut, getCudaStream(), std::to_string(itest++)+"faceInfor otputs");
// 	}
// 	std::cout << std::endl;
	cudaMemcpyAsync(resultBufferHost, modelIObuffers.back(), modelOutputBytes*vmodelInputes.size(), cudaMemcpyDeviceToHost, getCudaStream());
	exitIfCudaError(cudaStreamSynchronize(getCudaStream()));
	float* r = resultBufferHost;
	CFaceInfor* dst = (CFaceInfor*)voutput;
	int featLength = m_dimsOut.c()*m_dimsOut.h()*m_dimsOut.w();
//	std::cout << "featLength start ---- " << std::endl;
//	std::cout << featLength << std::endl;
//	std::cout << dst->rescore << std::endl;
//	std::cout << dst->blur << std::endl;
//	std::cout << "featLength over ---- " << std::endl;
	for (int i = 0; i < vmodelInputes.size(); ++i)
	{
		dst->m_pproducer = vmodelInputes[i];
		r += 512;
		if (featLength > 512) {dst->age = *r++;}
		if (featLength > 514) { dst->gender = (r[0] > r[1]); r++; r++;}
		if (featLength > 516) { dst->rescore = __softmax(r[1], r[0]); r++; r++;}
		if (featLength > 518) { dst->blur = __softmax(r[1], r[0]); r++; r++; }
		//xxs add reco first image  per time
		if (m_pfaceRecognization && i ==0)
		{
			static int ireco = 0;
			static int iigno = 0;
			m_faceAttriOI.front() = dst;
//			std::cout <<"dst"<<dst<< std::endl;
			m_faceAttriOI.back() = (CDataShared*)((char*)modelIObuffers.front()+i*modelInputBytes);
			CTimer timer0;
			timer0.start(1);
			m_pfaceRecognization->preProcessV(m_faceAttriOI);

			timer0.stop();
//			std::cout << "face reco pre times :" << timer0.getAvgMillisecond() << std::endl;
			if (m_faceAttriOI.front())
			{
				ireco++;
// 				std::cout << ireco << " Recognization" << std::endl;
 				CTimer timer1;
			    timer1.start(1);
				m_pfaceRecognization->inference(1);
				timer1.stop();
//				std::cout << "face rec  infer times :" << timer1.getAvgMillisecond() << std::endl;

				CTimer timer2;
			    timer2.start(1);
				m_pfaceRecognization->postProcessV(m_faceAttriOI, dst);
				m_pfaceRecognization->get_feature(batch_face_feature);
				timer2.stop();
//				std::cout << "face rec post times :" << timer2.getAvgMillisecond() << std::endl;
			}
			else {
				iigno++;
// 				std::cout << iigno << " ignorRecognization" << std::endl;
			}
		}
		dst++;
	}
}

float CFaceAttributeCalculator::__softmax(float x0, float x1)
{
	float ex0 = exp(x0);
	float ex1 = exp(x1);
	ex1 += ex0;
	return ex0 / ex1;
}

void CFaceAttributeCalculator::affine(int vbatchSize, float* vp68points, CImage* vimage, float* vdst)
{
	clock_t start, end;
	bool show_time = false;
	start = clock();
	affine_instance.stream = getViceCudaStream();
	affine_instance.show_eclipse_time = show_time;

	affine_instance.affineInterface(vbatchSize, vp68points, (unsigned char*)vimage->getClodData(true),
		vimage->width, vimage->height, vdst);
	end = clock();
	if (show_time)
		printf("Total execution time %3.4f ms\n", (double)(end - start) / CLOCKS_PER_SEC * 1000 / vbatchSize);
}
void CFaceAttributeCalculator::get_feature(std::vector<std::vector<float>>& fea)
{
    fea = batch_face_feature;
}
