/*
 * @brief: Demonstrates how to convert data from a multimap of vectors to a tensor
 * 			then builds a simple net that does binary classification 
 * 
 */


#include <list>
#include <algorithm>
#include <caffe2/core/init.h>
#include <caffe2/core/operator.h>
#include <caffe2/core/operator_gradient.h>

//https://github.com/leonardvandriel/caffe2_cpp_tutorial/blob/master/src/caffe2/binaries/mnist.cc
//https://github.com/caffe2/tutorials/blob/master/MNIST.ipynb
//https://caffe2.ai/docs/tutorial-toy-regression.html
//https://github.com/leonardvandriel/caffe2_cpp_tutorial/blob/master/src/caffe2/binaries/toy.cc



namespace caffe2 {

void print(Blob *blob, const std::string &name) {
	//auto tensor = blob->Get<TensorCPU>();
	Tensor* tensor = BlobGetMutableTensor(blob, caffe2::DeviceType::CPU);
	const auto &data = tensor->data<float>();
	std::cout << name << "(" << tensor->dims()
            << "): " << std::vector<float>(data, data + tensor->size())
						<< std::endl;
}


/*
 * @brief: Takes FnlData multimap, converts and splits it into a feature tensor and label tensor
 *
 */
std::vector<Tensor> FnlDataToTensor(std::multimap<boost::posix_time::ptime, std::vector<long double> > &data){

	int nClasses = 1;

	std::multimap<boost::posix_time::ptime, std::vector<long double>>::iterator cBegin = data.begin();
	std::multimap<boost::posix_time::ptime, std::vector<long double>>::iterator cEnd = data.end();

	int nFeatures = cBegin->second.size() -1 ;//-1 is because 1 column is the labels
	int nRows = what.size();

	std::vector<int> dim({nRows,nFeatures});

	Tensor featureTen(dim, caffe2::DeviceType::CPU);
	std::vector<int> dimLabel({nRows,nClasses});
	Tensor labelTen(nRows, caffe2::DeviceType::CPU);
	//Tensor labelTen(dimLabel, caffe2::DeviceType::CPU);

	int featureCount = 0;
	int labelCount = 0;
	int label = 0;
	std::vector<Tensor> featureLabelTens;

	for(; cBegin!=cEnd; cBegin++){
		auto vIt = cBegin->second.begin();
		auto vEnd = cBegin->second.end();
		vEnd--;
		for(; vIt!=vEnd; ++vIt)
		{
			featureTen.mutable_data<float>()[featureCount] =*vIt;
			cout<<"featureTen "<<featureTen.mutable_data<float >()[featureCount]<<endl;
			featureCount++;
		}
		label = (int)(std::trunc(*vIt));
		labelTen.mutable_data<int>()[labelCount] = label;//zero
		cout<<"labelTen "<<labelTen.mutable_data<int>()[labelCount]<<endl;

	}
	featureLabelTens.push_back(featureTen);
	featureLabelTens.push_back(labelTen);

	return featureLabelTens;

}


void run1(std::multimap<boost::posix_time::ptime, std::vector<long double>> &data) {
	std::cout << std::endl;
	std::cout << "## Caffe2 Intro Tutorial ##" << std::endl;
	std::cout << "https://caffe2.ai/docs/intro-tutorial.html" << std::endl;
	std::cout << std::endl;

	// >>> from caffe2.python import workspace, model_helper
	// >>> import numpy as np
	Workspace workspace;


	std::vector<Tensor> featureLabelTens;
	featureLabelTens = FnlDataToTensor(data);


	// >>> workspace.FeedBlob("data", data)
	{
		Blob* myBlob = workspace.CreateBlob("data");
		Tensor* tensor = caffe2::BlobGetMutableTensor(myBlob, caffe2::DeviceType::CPU);
		tensor->CopyFrom(featureLabelTens[0]);
		cout<<"Feature Tensor Info "<<tensor->DebugString()<<endl;
	}

	// >>> workspace.FeedBlob("label", label)
	{
		Blob* myBlob = workspace.CreateBlob("label");
		Tensor* tensor = caffe2::BlobGetMutableTensor(myBlob, caffe2::DeviceType::CPU);
		tensor->CopyFrom(featureLabelTens[1]);
		cout<<"Label Tensor Info "<<tensor->DebugString()<<endl;


	}



	// >>> m = model_helper.ModelHelper(name="my first net")
	NetDef initModel;
	initModel.set_name("my first net_init");
	NetDef predictModel;
	predictModel.set_name("my first net");

	// >>> ONE = init_net.ConstantFill([], "ONE", shape=[1], value=1.)
	{
		auto op = initModel.add_op();
		op->set_type("ConstantFill");
		auto arg1 = op->add_arg();
		arg1->set_name("shape");
		arg1->add_ints(1);
		auto arg2 = op->add_arg();
		arg2->set_name("value");
		arg2->set_f(1.0);
		op->add_output("ONE");
	}

	// >>> ITER = init_net.ConstantFill([], "ITER", shape=[1], value=0,
	// dtype=core.DataType.INT32)
	{
		auto op = initModel.add_op();
		op->set_type("ConstantFill");
		auto arg1 = op->add_arg();
		arg1->set_name("shape");
		arg1->add_ints(1);
		auto arg2 = op->add_arg();
		arg2->set_name("value");
		arg2->set_i(0);
		auto arg3 = op->add_arg();
		arg3->set_name("dtype");
		arg3->set_i(TensorProto_DataType_INT32);
		op->add_output("ITER");
	}

	// >>> weight = m.param_initModel.XavierFill([], 'fc_w', shape=[10, 100])
	{
		auto op = initModel.add_op();
		op->set_type("UniformFill");
		auto arg1 = op->add_arg();
		arg1->set_name("shape");
		arg1->add_ints(2);//This seems to be tied to the number of classes so 2 = binary
		arg1->add_ints(9);//This is the number of features
		auto arg2 = op->add_arg();
		arg2->set_name("min");
		float min = -std::sqrt(6.0/(9+2));
		arg2->set_f(min);//arg2->set_f(-1);
		auto arg3 = op->add_arg();
		arg3->set_name("max");
		float max = std::sqrt(6.0/(9+2));
		arg3->set_f(max);
		op->add_output("fc_w");
	}


	// >>> bias = m.param_initModel.ConstantFill([], 'fc_b', shape=[10, ])
	{
		auto op = initModel.add_op();
		op->set_type("ConstantFill");
		auto arg = op->add_arg();
		arg->set_name("shape");
		arg->add_ints(2);//This seems to be tied to the number of classes so 2 = binary
		op->add_output("fc_b");
	}

	std::vector<OperatorDef*> gradient_ops;

	// >>> fc_1 = m.net.FC(["data", "fc_w", "fc_b"], "fc1")
	{
		auto op = predictModel.add_op();
		op->set_type("FC");
		op->add_input("data");
		op->add_input("fc_w");
		op->add_input("fc_b");
		op->add_output("fc1");
		gradient_ops.push_back(op);
	}

	// >>> pred = m.net.Sigmoid(fc_1, "pred")
	{
		auto op = predictModel.add_op();
		op->set_type("Sigmoid");
		op->add_input("fc1");
		op->add_output("pred");
		gradient_ops.push_back(op);
	}

	// >>> [softmax, loss] = m.net.SoftmaxWithLoss([pred, "label"], ["softmax",
	// "loss"])
	{
		auto op = predictModel.add_op();
		op->set_type("SoftmaxWithLoss");
		op->add_input("pred");
		op->add_input("label");
		op->add_output("softmax");
		op->add_output("loss");
		gradient_ops.push_back(op);
	}

	{
		auto op = predictModel.add_op();
		op->set_type("Accuracy");
		op->add_input("softmax");//op->add_input("ypred");
		op->add_input("label");
		op->add_output("accuracy");

	}

	{
		auto op = predictModel.add_op();
		op->set_type("ArgMax");
		op->add_input("softmax");//op->add_input("ypred");
		op->add_output("argmax");

	}

	// >>> m.AddGradientOperators([loss])
	{
		auto op = predictModel.add_op();
		op->set_type("ConstantFill");
		auto arg = op->add_arg();
		arg->set_name("value");
		arg->set_f(1.0);
		op->add_input("loss");
		op->add_output("loss_grad");
		op->set_is_gradient_op(true);
	}
	std::reverse(gradient_ops.begin(), gradient_ops.end());
	for (auto op : gradient_ops) {
		vector<GradientWrapper> output(op->output_size());
		for (auto i = 0; i < output.size(); i++) {
			output[i].dense_ = op->output(i) + "_grad";
		}
		GradientOpsMeta meta = GetGradientForOp(*op, output);
		auto grad = predictModel.add_op();
		grad->CopyFrom(meta.ops_[0]);
		grad->set_is_gradient_op(true);
	}

	// >>> train_net.Iter(ITER, ITER)
	{
		auto op = predictModel.add_op();
		op->set_type("Iter");
		op->add_input("ITER");
		op->add_output("ITER");
	}

	// >>> LR = train_net.LearningRate(ITER, "LR", base_lr=-0.1, policy="step",
	// stepsize=20, gamma=0.9)
	{
		auto op = predictModel.add_op();
		op->set_type("LearningRate");
		auto arg1 = op->add_arg();
		arg1->set_name("base_lr");
		arg1->set_f(-0.00004);
		auto arg2 = op->add_arg();
		arg2->set_name("policy");
		arg2->set_s("step");
		auto arg3 = op->add_arg();
		arg3->set_name("stepsize");
		arg3->set_i(20);
		auto arg4 = op->add_arg();
		arg4->set_name("gamma");
		arg4->set_f(0.9);
		op->add_input("ITER");
		op->add_output("LR");
	}

	// >>> train_net.WeightedSum([W, ONE, gradient_map[W], LR], W)
	{
		auto op = predictModel.add_op();
		op->set_type("WeightedSum");
		op->add_input("fc_w");
		op->add_input("ONE");
		op->add_input("fc_w_grad");
		op->add_input("LR");
		op->add_output("fc_w");
	}

	// >>> train_net.WeightedSum([B, ONE, gradient_map[B], LR], B)
	{
		auto op = predictModel.add_op();
		op->set_type("WeightedSum");
		op->add_input("fc_b");
		op->add_input("ONE");
		op->add_input("fc_b_grad");
		op->add_input("LR");
		op->add_output("fc_b");
	}

	// >>> workspace.RunNetOnce(m.param_init_net)
	CAFFE_ENFORCE(workspace.RunNetOnce(initModel));


	// >>> workspace.CreateNet(m.net)
	CAFFE_ENFORCE(workspace.CreateNet(predictModel));

	print(workspace.GetBlob("fc_w"), "fc_w Before Training");


	// >>> for j in range(0, 100):
	for (auto i = 0; i < 500; i++) {
		CAFFE_ENFORCE(workspace.RunNet(predictModel.name()));

		if(i%100  == 0){
			std::cout << "step: " << i << " loss: ";
					print(workspace.GetBlob("loss"),"loss");
					std::cout << std::endl;
		}
	}

	// >>> print(workspace.FetchBlob("softmax"))
	print(workspace.GetBlob("softmax"), "softmax");
	print(workspace.GetBlob("accuracy"), "accuracy");
	auto myAcc= workspace.GetBlob("accuracy")->Get<Tensor>();

	auto softmaxTen = workspace.GetBlob("softmax")->Get<Tensor>();
	cout<<"softmaxTen.size() "<<softmaxTen.size()<<endl;

	auto myArgmax = workspace.GetBlob("argmax")->Get<Tensor>();
	cout<<"argmax "<<myArgmax.data<long>()[0]<<" "<<myArgmax.DebugString()<<endl;

	for( int i = 0; i< myArgmax.size(); ++i)
		cout<<"argmax value "<<myArgmax.data<long>()[i]<<" softmax "<<softmaxTen.data<float>()[i]<<endl;

	std::cout << std::endl;

	print(workspace.GetBlob("fc_w"), "fc_w After Training");


	// >>> print(workspace.FetchBlob("loss"))
	print(workspace.GetBlob("loss"), "loss");

	print(workspace.GetBlob("pred"), "pred");
}

} // namespace caffe2


int main(int argc, char **argv) {

	caffe2::GlobalInit(&argc, &argv);
  

	std::vector<std::multimap<boost::posix_time::ptime, std::vector<long double> > > data;

	auto FnlData = BuildFnlData(data);


	caffe2::run1(FnlData);
	google::protobuf::ShutdownProtobufLibrary();


	return 0;
}





























