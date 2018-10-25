/*
 * asdf.cpp
 *
 *  Created on: Oct 7, 2018
 *      Author: ryan
 */

  // namespace caffe2

#include "ReadWriteNet.h"

#include <list>
#include <caffe2/proto/caffe2_pb.h>

#include <caffe2/core/init.h>
#include <caffe2/core/operator.h>
#include <caffe2/core/operator_gradient.h>
#include <caffe2/utils/proto_utils.h>



namespace caffe2 {

void print(Blob* blob, const std::string& name) {
	//auto tensor = blob->->Get<TensorCPU>();
	Tensor* tensor = BlobGetMutableTensor(blob, caffe2::DeviceType::CPU);
	const auto& data = tensor->data<float>();
	std::cout << name << "(" << tensor->dims()
            				<< "): " << std::vector<float>(data, data + tensor->size())
							<< std::endl;
}

void run() {
	std::cout << std::endl;
	std::cout << "## Caffe2 Intro Tutorial ##" << std::endl;
	std::cout << "https://caffe2.ai/docs/intro-tutorial.html" << std::endl;
	std::cout << std::endl;

	// >>> from caffe2.python import workspace, model_helper
	// >>> import numpy as np
	Workspace workspace;



	// >>> x = np.random.rand(4, 3, 2)
	std::vector<float> x(4 * 3 * 2);
	//for (auto& v : x) {
	//	v = (float)rand() / RAND_MAX;
	//}

	int count = 0;
	Tensor value = Tensor(x.size(), caffe2::DeviceType::CPU);
	for (auto &v: x) {
		v = (float)rand() / RAND_MAX;
		value.mutable_data<float>()[count]=v;
		count++;
	}

	// >>> print(x)
	std::cout << x << std::endl;

	// >>> workspace.FeedBlob("my_x", x)
	{
		Blob* my_xBlob = workspace.CreateBlob("my_x");
		Tensor* tensor = BlobGetMutableTensor(my_xBlob, caffe2::DeviceType::CPU);


		//tensor->ResizeLike(value); //still work
		//tensor->ShareData(value); //still work
		tensor->CopyFrom(value);
	}

	// >>> x2 = workspace.FetchBlob("my_x")
	// >>> print(x2)
	{
		Blob* blob = workspace.GetBlob("my_x");
		print(blob, "my_x");
	}


	/*
	 * the above just demonstrated how to get data data into a tensor thats in a blob which is in a workspace
	 */
	// >>> data = np.random.rand(16, 100).astype(np.float32)


	//std::vector<float> data(16 * 100);
	std::vector<float> data(16*10);
	std::vector<int> dim({16,10});
	count = 0;
	/*********************************************/
	/*             Important note
	 * From what I could gather the data in the tensor is stored as a one dimensional array
	 * so if you have this
	 *
	 * a	b	c	d	e	f
	 * g	h	I	j	k	l
	 * m	n	o	p	k	r
	 *
	 * you feed it into the Tensor something like this
	 *
	 * r
	 * p
	 * o
	 * n
	 * m
	 * l
	 * k
	 * k
	 * j
	 * I
	 * h
	 * g
	 * f
	 * e
	 * d
	 * c
	 * b
	 * a
	 *
	 * then the dimensions tell the tensor how the data is should look, so in this example
	 * the dimensions would be {3,6}
	 *
	 * This was not obvious to me at first
	 *
	 * */
	Tensor dataTen(dim, caffe2::DeviceType::CPU);

	for (auto& v : data) {
			v = (float)rand() / RAND_MAX;
			dataTen.mutable_data<float>()[count] =v;
		count++;
	}

	//just to show that the data is there
	for(int a =0; a<count; ++a)
		cout<<dataTen.mutable_data<float >()[a]<<endl;
	cout<<dataTen.DebugString()<<endl;

	// >>> label = (np.random.rand(16) * 10).astype(np.int32)
	std::vector<int> label(16,1);
	count = 0;
	Tensor labelTen = Tensor(label.size(), caffe2::DeviceType::CPU);
	for (auto& v : label) {
		v = rand() %10;
		labelTen.mutable_data<int>()[count]=v;
		count++;
	}



	// >>> workspace.FeedBlob("data", data)
	{
		//auto tensor = workspace.CreateBlob("data")->GetMutable<TensorCPU>();
		Blob* myBlob = workspace.CreateBlob("data");
		Tensor* tensor = caffe2::BlobGetMutableTensor(myBlob, caffe2::DeviceType::CPU);

		//auto value = TensorCPU({16, 100}, data, NULL);
		//dataTen is used instead of value

		//tensor->ResizeLike(value);
		//tensor->ShareData(value);
		tensor->CopyFrom(dataTen);//the above two lines works this is just a different way to do it you will see later that I do it this way
	}

	// >>> workspace.FeedBlob("label", label)
	{
		//auto tensor = workspace.CreateBlob("label")->GetMutable<TensorCPU>();
		Blob* myBlob = workspace.CreateBlob("label");
		Tensor* tensor = caffe2::BlobGetMutableTensor(myBlob, caffe2::DeviceType::CPU);

		//auto value = TensorCPU({16}, label, NULL);
		//labelTen is used instead of value

		//tensor->ResizeLike(value);
		//tensor->ShareData(value);
		tensor->CopyFrom(labelTen);//the above two lines works this is just a different way to do it you will see later that I do it this way

	}

	// >>> m = model_helper.ModelHelper(name="my first net")
	NetDef initModel;
	initModel.set_name("my first net_init");
	NetDef predictModel;
	predictModel.set_name("my first net");

	// >>> weight = m.param_initModel.XavierFill([], 'fc_w', shape=[10, 100])
	{
		auto op = initModel.add_op();
		op->set_type("XavierFill");
		auto arg = op->add_arg();
		arg->set_name("shape");
		arg->add_ints(16);
		arg->add_ints(10);
		op->add_output("fc_w");
	}


	// >>> bias = m.param_initModel.ConstantFill([], 'fc_b', shape=[10, ])
	{
		auto op = initModel.add_op();
		op->set_type("ConstantFill");
		auto arg = op->add_arg();
		arg->set_name("shape");
		arg->add_ints(16);
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

	// >>> print(str(m.net.Proto()))
	// std::cout << std::endl;
	// print(predictModel);

	// >>> print(str(m.param_init_net.Proto()))
	// std::cout << std::endl;
	// print(initModel);

	// >>> workspace.RunNetOnce(m.param_init_net)
	CAFFE_ENFORCE(workspace.RunNetOnce(initModel));


	// >>> workspace.CreateNet(m.net)
	CAFFE_ENFORCE(workspace.CreateNet(predictModel));


	// >>> for j in range(0, 100):
	for (auto i = 0; i < 100; i++) {
		// >>> data = np.random.rand(16, 100).astype(np.float32)
		//std::vector<float> data(16 * 100);
		std::vector<float> data(16*10);
		count=0;
		for (auto& v : data) {
			v = (float)rand() / RAND_MAX;
			dataTen.mutable_data<float>()[count] = v;
			count++;
		}

		// >>> label = (np.random.rand(16) * 10).astype(np.int32)
		std::vector<int> label(16);
		count = 0;
		for (auto& v : label) {
			v = rand() %10;
			labelTen.mutable_data<int>()[count] = v;
			count++;

		}

		// >>> workspace.FeedBlob("data", data)
		{
			//auto tensor = workspace.GetBlob("data")->GetMutable<TensorCPU>();
			Blob* myBlob = workspace.GetBlob("data");
			Tensor* tensor = caffe2::BlobGetMutableTensor(myBlob, caffe2::DeviceType::CPU);
			//auto value = TensorCPU({16, 100}, data, NULL);
			//tensor->ShareData(value);
			tensor->ResizeLike(dataTen);
			tensor->ShareData(dataTen);


		}

		// >>> workspace.FeedBlob("label", label)
		{
			//auto tensor = workspace.GetBlob("label")->GetMutable<TensorCPU>();
			Blob* myBlob = workspace.GetBlob("label");
			Tensor* tensor = caffe2::BlobGetMutableTensor(myBlob, caffe2::DeviceType::CPU);
			//auto value = TensorCPU({16}, label, NULL);
			//tensor->ShareData(value);
			tensor->ResizeLike(labelTen);
			tensor->ShareData(labelTen);

		}

		cout<<predictModel.DebugString()<<endl;
		cout<<predictModel.external_input_size()<<endl;
		predictModel.InitAsDefaultInstance();


		// >>> workspace.RunNet(m.name, 10)   # run for 10 times
		for (auto j = 0; j < 10; j++) {
			predictModel.CheckInitialized();
			CAFFE_ENFORCE(workspace.RunNet(predictModel.name()));
			 std::cout << "step: " << i << " loss: ";
			 print(workspace.GetBlob("loss"),"loss");
			 std::cout << std::endl;
		}
	}

	std::cout << std::endl;

	// >>> print(workspace.FetchBlob("softmax"))
	print(workspace.GetBlob("softmax"), "softmax");

	std::cout << std::endl;

	// >>> print(workspace.FetchBlob("loss"))
	print(workspace.GetBlob("loss"), "loss");
}

}
int main(int argc, char **argv) {

	caffe2::GlobalInit(&argc, &argv);
	caffe2::run();
	google::protobuf::ShutdownProtobufLibrary();


	return 0;
}
