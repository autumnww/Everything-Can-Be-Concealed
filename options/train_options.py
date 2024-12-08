from .base_options import BaseOptions


class TrainOptions(BaseOptions):
    """This class includes training options.

    It also includes shared options defined in BaseOptions.
    """

    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)


        parser.add_argument('--display_freq', type=int, default=400, help='frequency of showing training results on screen')
        parser.add_argument('--display_id', type=int, default=1, help='window id of the web display')
        parser.add_argument('--display_server', type=str, default="http://localhost", help='visdom server of the web display')
        parser.add_argument('--display_env', type=str, default='main', help='visdom display environment name (default is "main")')
        parser.add_argument('--display_port', type=int, default=8088, help='visdom port of the web display')
        parser.add_argument('--update_html_freq', type=int, default=500, help='frequency of saving training results to html')
        parser.add_argument('--print_freq', type=int, default=300, help='frequency of showing training results on console')
        parser.add_argument('--no_html', action='store_true', help='do not save intermediate training results to [opt.checkpoints_dir]/[opt.name]/web/')
        # network saving and loading parameters
        parser.add_argument('--save_latest_freq', type=int, default=5000, help='frequency of saving the latest results')
        parser.add_argument('--save_epoch_freq', type=int, default=1, help='frequency of saving checkpoints at the end of epochs')
        parser.add_argument('--save_by_iter', action='store_true', help='whether saves model by iteration')
        parser.add_argument('--continue_train', action='store_true', help='continue training: load the latest model')
        parser.add_argument('--epoch_count', type=int, default=1, help='the starting epoch count, we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>, ...')
        parser.add_argument('--phase', type=str, default='train', help='train, val, test, etc')
        # training parameters
        parser.add_argument('--niter', type=int, default=100, help='# of iter at starting learning rate')
        parser.add_argument('--niter_decay', type=int, default=100, help='# of iter to linearly decay learning rate to zero')
        parser.add_argument('--beta1', type=float, default=0.5, help='momentum term of adam')
        parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate for adam')  
        parser.add_argument('--split_num', type=int, default=8, help='spilt image to num patchs')
        parser.add_argument('--use_dist', type=bool, default=False, help='weather to use dist')
        # hyper-parameters
        parser.add_argument('--gan_mode', type=str, default='wgangp', help='the type of GAN objective. [vanilla| lsgan | wgangp]. vanilla GAN loss is the cross-entropy objective used in the original GAN paper.')
        parser.add_argument('--lambda_g', type=float, default=1.0, help='weight for adversarial loss')
        parser.add_argument('--lambda_style', type=float, default=1.0, help='weight for style loss')
        parser.add_argument('--lambda_content', type=float, default=1.0, help='weight for content loss')
        parser.add_argument('--lambda_tv', type=float, default=1e-5, help='weight for smooth loss')
        parser.add_argument('--lambda_mask', type=float, default=1, help='weight for mask rec loss')

        self.isTrain = True
        return parser