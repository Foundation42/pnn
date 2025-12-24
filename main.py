#!/usr/bin/env python3
"""
Physical Neural Network PCB Compiler

Train neural networks where the physical PCB layout is part of the optimization.
Watch the network "condense" into an optimal geometric form that can be fabricated.

Usage:
    python main.py              # Train and generate animation
    python main.py --epochs N   # Train for N epochs
    python main.py --no-anim    # Skip animation generation
"""

import argparse
from train import train_xor, save_model
from visualize import (
    create_condensation_animation,
    plot_final_layout,
    plot_training_progress
)


def main():
    parser = argparse.ArgumentParser(
        description='Physical Neural Network PCB Compiler',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python main.py                    # Full training + visualization
    python main.py --epochs 10000     # Longer training
    python main.py --layout-weight 0.2 # Stronger layout optimization
        """
    )

    parser.add_argument('--epochs', type=int, default=5000,
                        help='Number of training epochs (default: 5000)')
    parser.add_argument('--layout-weight', type=float, default=0.1,
                        help='Weight for layout loss (default: 0.1)')
    parser.add_argument('--hidden-size', type=int, default=4,
                        help='Number of hidden neurons (default: 4)')
    parser.add_argument('--board-size', type=int, nargs=2, default=[50, 50],
                        help='Board dimensions in mm (default: 50 50)')
    parser.add_argument('--no-anim', action='store_true',
                        help='Skip animation generation')
    parser.add_argument('--fps', type=int, default=30,
                        help='Animation FPS (default: 30)')
    parser.add_argument('--duration', type=float, default=10.0,
                        help='Animation duration in seconds (default: 10)')
    parser.add_argument('--output-prefix', type=str, default='pnn',
                        help='Prefix for output files (default: pnn)')

    args = parser.parse_args()

    print()
    print("=" * 60)
    print("  Physical Neural Network PCB Compiler")
    print("  Matter condensing from the training loop...")
    print("=" * 60)
    print()

    # Train the network
    model, history = train_xor(
        epochs=args.epochs,
        layout_weight=args.layout_weight,
        hidden_size=args.hidden_size,
        board_size=tuple(args.board_size),
        verbose=True
    )

    # Save model
    model_path = f"{args.output_prefix}_model.pt"
    save_model(model, model_path)

    # Generate visualizations
    print("\n" + "=" * 60)
    print("  Generating Visualizations")
    print("=" * 60 + "\n")

    # Final layout
    layout_path = f"{args.output_prefix}_layout.png"
    plot_final_layout(model, save_path=layout_path)

    # Training progress
    progress_path = f"{args.output_prefix}_progress.png"
    plot_training_progress(history, save_path=progress_path)

    # Condensation animation
    if not args.no_anim:
        anim_path = f"{args.output_prefix}_condensation.gif"
        input_pos = model.input_positions.numpy()
        output_pos = model.output_positions.numpy()

        create_condensation_animation(
            history,
            board_size=(model.board_width, model.board_height),
            input_positions=input_pos,
            output_positions=output_pos,
            save_path=anim_path,
            fps=args.fps,
            duration=args.duration
        )

    print("\n" + "=" * 60)
    print("  Complete!")
    print("=" * 60)
    print(f"\nOutput files:")
    print(f"  Model:           {model_path}")
    print(f"  Layout:          {layout_path}")
    print(f"  Training curves: {progress_path}")
    if not args.no_anim:
        print(f"  Animation:       {anim_path}")

    print("\nNext steps:")
    print("  1. Review the animation to see the condensation process")
    print("  2. Check the layout to see the final PCB design")
    print("  3. Run export_kicad.py to generate manufacturing files")
    print()


if __name__ == "__main__":
    main()
