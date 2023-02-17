from tfx import v1 as tfx


def get_pusher(trainer, serving_model_dir, evaluator) -> tfx.components.Pusher:
    pusher = tfx.components.Pusher(
        model=trainer.outputs['model'],
        model_blessing=evaluator.outputs['blessing'],
        push_destination=tfx.proto.PushDestination(
            filesystem=tfx.proto.PushDestination.Filesystem(
                base_directory=serving_model_dir)))
    return pusher
