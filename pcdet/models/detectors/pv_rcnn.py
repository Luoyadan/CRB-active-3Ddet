from .detector3d_template import Detector3DTemplate


class PVRCNN(Detector3DTemplate):
    def __init__(self, model_cfg, num_class, dataset):
        super().__init__(model_cfg=model_cfg, num_class=num_class, dataset=dataset)
        self.module_list = self.build_networks()

    def forward(self, batch_dict):
        for cur_module in self.module_list:
            batch_dict = cur_module(batch_dict)

        if self.training:
            loss, tb_dict, disp_dict = self.get_training_loss()

            ret_dict = {
                'loss': loss,
                'rcnn_reg_gt': self.roi_head.forward_ret_dict['rcnn_reg_gt'],
                'rcnn_cls_gt': self.roi_head.forward_ret_dict['rcnn_cls_labels'],
                'rcnn_cls': batch_dict['rcnn_cls'],
                'rcnn_reg': batch_dict['rcnn_reg'],
                'rpn_preds': batch_dict['rpn_preds']
            }
            return ret_dict, tb_dict, disp_dict
        else:
            pred_dicts, recall_dicts = self.post_processing(batch_dict)
            return pred_dicts, recall_dicts

    def get_training_loss(self):
        disp_dict = {}
        lal_flag = hasattr(self.roi_head, 'loss_net')
        if lal_flag:
            # if requires grad -> training loss net
            lal_flag = list(self.roi_head.loss_net.children())[0].weight.requires_grad
        loss_rpn, tb_dict = self.dense_head.get_loss(reduce=not lal_flag)
        loss_point, tb_dict = self.point_head.get_loss(tb_dict, reduce=not lal_flag)
        loss_rcnn, tb_dict = self.roi_head.get_loss(tb_dict, reduce=not lal_flag)
        loss = loss_rpn + loss_point + loss_rcnn
        if lal_flag:
            loss_loss_net = self.roi_head.get_loss_loss_net(tb_dict, loss)
            loss = loss.mean()
            loss += loss_loss_net
        return loss, tb_dict, disp_dict
