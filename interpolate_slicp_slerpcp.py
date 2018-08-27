
"""
This file creates SLI-CP and Slerp-CP interpolations using Latent Blending. 
Before using this file, use train_pixelvae.py to train
a PixelVAE on CIFAR-10 (or another dataset). This file is currently
set to run for CIFAR-10; however, the lines which need
to be adjusted in order to run this file on another dataset have been labelled.

When using a different set of parameters or PixelVAE architecture, 
change the sampling_loop file to the one 
which is tailored to run on your desired set of parameters.

This code is adapted from: https://github.com/igul222/PixelVAE

The PixelVAE was initially published in: 

PixelVAE: A Latent Variable Model for Natural Images
Ishaan Gulrajani, Kundan Kumar, Faruk Ahmed, Adrien Ali Taiga, 
Francesco Visin, David Vazquez, Aaron Courville
"""

# Import all the libraries needed
# .......

"""

NOTE: ALL LINES UNTIL THE ONE AT WHICH THIS FILE BEGINS BELOW ARE THE SAME AS IN
'interpolate_sli_slerp_mixup.py'. Thus, we have excluded them from the print version
of our code. Our Github repository contains the full version of this file.

"""

    # CREATE MIXED EXAMPLES

    if MODE == 'one_level': # Other options - those for multiple-level PixelVAEs - have been pruned, as we did not use them in our final analyses

        ch_sym = tf.placeholder(tf.int32, shape=None)
        y_sym = tf.placeholder(tf.int32, shape=None)
        x_sym = tf.placeholder(tf.int32, shape=None)
        logits = tf.reshape(tf.slice(outputs1, tf.stack([0, ch_sym, y_sym, x_sym, 0]), tf.stack([-1, 1, 1, 1, -1])), [-1, 256])
        dec1_fn_out = tf.multinomial(logits, 1)[:, 0]
          
        def dec1_fn(_latents, _targets, _ch, _y, _x):
            return session.run(dec1_fn_out, feed_dict={latents1: _latents, images: _targets, ch_sym: _ch, y_sym: _y, x_sym: _x, total_iters: 99999, bn_is_training: False, bn_stats_iter:0})
        def enc_fn(_images):
            return session.run(latents1, feed_dict={images: _images, total_iters: 99999, bn_is_training: False, bn_stats_iter:0})

        def generate_and_save_samples(tag):
            from keras.utils import np_utils
            import itertools
            
            # Create placeholer arrays which will hold mixed examples and their labels (separate arrays for SLI-CP and Slerp-CP)
            x_augmentation_set_slicp = np.zeros((1, N_CHANNELS, HEIGHT, WIDTH)) 
            y_augmentation_set_slicp = np.zeros((1, 1, NUM_CLASSES))
            x_augmentation_set_slerpcp = np.zeros((1, N_CHANNELS, HEIGHT, WIDTH)) 
            y_augmentation_set_slerpcp = np.zeros((1, 1, NUM_CLASSES))
            
            ### Find the most closely related class pairs using the L2 distance between images in the training set
            (x_train_set, y_train_set), (x_test_set, y_test_set) = cifar10.load_data()
            x_train_set = x_train_set.transpose(0,3,1,2)
            x_test_set = x_test_set.transpose(0,3,1,2)
            seed = 333
            x_train_set, x_dev_set, y_train_set, y_dev_set = train_test_split(x_train_set, y_train_set, test_size=0.1, random_state=seed)
            
            all_latents = np.zeros((1,LATENT_DIM_2)).astype('float32') 
            x_train_set = x_train_set.reshape(-1, N_CHANNELS, HEIGHT, WIDTH)
            y_train_set = y_train_set.reshape(-1, 1)
         
            # Encode all images
            print "Encoding images"
            for j in range(x_train_set.shape[0]):
               latestlatents = enc_fn(x_train_set[j,:].reshape(1, N_CHANNELS, HEIGHT, WIDTH))
               latestlatents = latestlatents.reshape(-1,LATENT_DIM_2)
               all_latents = np.concatenate((all_latents, latestlatents), axis=0)
        
            all_latents = np.delete(all_latents, (0), axis=0)
         
            # Find the latent mean codes by class
            print "Finding class means"
            classmeans = np.zeros((NUM_CLASSES, LATENT_DIM_2)).astype('float32')
            for k in range(NUM_CLASSES):
               idk = np.asarray(np.where(np.equal(y_train_set,k))[0])
               all_latents_groupk = all_latents[idk,:]
               classmeans[k,:] = np.mean(all_latents_groupk, axis=0)
      
            # Create a list containing all pairs of classes
            print "Finding pairs"
            pairs = np.array(list(itertools.combinations(range(NUM_CLASSES),2)))
            num_pairs = pairs.shape[0]
         
            # Find the L2 distance between the members of each pair of classes
            meandist = np.zeros((num_pairs)).astype('float64')
            classarray = np.arange(NUM_CLASSES)
            for m in range(num_pairs):
                  
                  # Find mean latent for images in the first class of the current pair
                  aidx = np.asarray(np.where(np.equal(classarray,pairs[m,0])))
                  a = np.asarray(classmeans[aidx,:])
                  
                  # Find mean latent for images in the second class of the current pair
                  bidx = np.asarray(np.where(np.equal(classarray,pairs[m,1])))
                  b = np.asarray(classmeans[bidx,:])
                  a = a.reshape(1, LATENT_DIM_2)
                  b = b.reshape(1, LATENT_DIM_2)
                  
                  # Find the L2 distance between theclass means
                  meandist[m] = np.linalg.norm(a-b)
            
            # Sort the distances between pairs and find the smallest distance. We have included code to interpolate between just the two closest classes. However, it is easy to extend this to more pairs (e.g., 3, as done in the final paper) by taking the later values of 'sorteddistances'. The commented-out lines below indicate how this could be done.
            sorteddistances = np.sort(meandist)
            closestdistance = sorteddistances[0]
            #secondclosestdistance = sorteddistances[1]
            #thirdclosestdistance = sorteddistances[2]
      
            # Draw out the pairs corresponding to these distances
            closestidx = np.asarray(np.where(np.equal(meandist, closestdistance))[0])
            #secondclosestidx = np.asarray(np.where(np.equal(meandist, secondclosestdistance))[0])
            #thirdclosestidx = np.asarray(np.where(np.equal(meandist, thirdclosestdistance))[0])

            closestpair = pairs[closestidx,:]
            #secondclosestpair = pairs[secondclosestidx,:]
            #thirdclosestpair = pairs[thirdclosestidx,:]
            
            
            ### Now that we have identified the closest-related pairs, we can conduct SLI-CP and Slerp-CP
            # The code below is set up to create interpolations between the closest two classes (for the sake of brevity).
            # The commented-out lines indicate how this can be adjusted to interpolate between greater numbers of closely-related classes.
            classpairs = closestpair
            #classpairs = np.concatenate((closestpair, secondclosestpair, thirdclosestpair), axis=0)
             
            # Function to translate numeric images into plots
            def color_grid_vis(X, nh, nw, save_path):
                # Original code from github.com/Newmu
                X = X.transpose(0,2,3,1)
                h, w = X[0].shape[:2]
                img = np.zeros((h*nh, w*nw, 3))
                for n, x in enumerate(X):
                    j = n/nw
                    i = n%nw
                    img[j*h:j*h+h, i*w:i*w+w, :] = x
                imsave(OUT_DIR + '/' + save_path, img)
                
            # This line controls how many images will be generated
            numsamples = 1125
               
            x_train_set_array = np.array(x_train_set)
            y_train_set_array = np.array(y_train_set)  
            
            for imagenum in range(numsamples):
               
               # Sample unique image indices from class pairs. Images will be interpolated in pairs. Pairs are listed in order.
               classindices = classpairs
               idx1 = np.asarray(np.where(np.equal(classindices[0,0],y_train_set))[0])
               idx2 = np.asarray(np.where(np.equal(classindices[0,1],y_train_set))[0])
                
               # Draw out the images in the training set corresponding to the two classes in the closest-related pair
               x_trainsubset1 = x_train_array[idx1,:]
               x_trainsubset2 = x_train_array[idx2,:]
               y_trainsubset1 = y_train_array[idx1,:]
               y_trainsubset2 = y_train_array[idx2,:]
               
               x_trainsubset1 = x_trainsubset1.reshape(-1, N_CHANNELS, HEIGHT, WIDTH)
               x_trainsubset2 = x_trainsubset2.reshape(-1, N_CHANNELS, HEIGHT, WIDTH)
               y_trainsubset1 = y_trainsubset1.reshape(-1, N_CHANNELS, HEIGHT, WIDTH)
               y_trainsubset2 = y_trainsubset2.reshape(-1, N_CHANNELS, HEIGHT, WIDTH)
               
               # Sample an image index from each of these classes
               imageindex1 = random.sample(range(x_trainsubset1.shape[0]),1)
               imageindex2 = random.sample(range(x_trainsubset2.shape[0]),1)
               
               # Draw out the corresponding images and labels
               image1 = x_trainsubset1[imageindex1,:]
               image2 = x_trainsubset2[imageindex2,:]
               label1 = y_trainsubset1[imageindex1,:]
               label2 = y_trainsubset2[imageindex2,:]
               
               # Reshape
               image1 = image1.reshape(1, N_CHANNELS, HEIGHT, WIDTH)
               image2 = image2.reshape(1, N_CHANNELS, HEIGHT, WIDTH)
               label1 = label1.reshape(1, 1)
               label2 = label2.reshape(1, 1)
                    
               # Save the original images
               print "Saving original samples"
               color_grid_vis(image1,1,1, 
                              'original_1_classes{}and{}_num{}.png'.format(class1,
                                                                           class2,imagenum))
               color_grid_vis(image2,1,1,
                              'original_2_classes{}and{}_num{}.png'.format(class1,
                                                                           class2,imagenum))
                      
               # Encode the images
               image_code1 = enc_fn(image1)
               image_code2 = enc_fn(image2)
               
               # Change labels to matrix form before performing interpolations
               label1 = np_utils.to_categorical(label1, NUM_CLASSES) 
               label2 = np_utils.to_categorical(label2, NUM_CLASSES) 
               
               # Lambda values to use for the specific weighting scheme. We use "p" instead of lambda in the code as it is shorter.
                  
               # This option is for constant lambda in {0.2, 0.4, 0.6, 0.8}
               pvals = np.linspace(0.2, 0.8, num=4) 
                  
               # This option is for Beta distributed lambda. Adjust the alpha values (first two parameters in the expression below) and number of samples to draw (third parameter in the expression below) based on the desired interpolation scheme.
               # pvals = np.random.beta(0.2, 0.2, 4) 
                    
               # Find angle between the two latent codes (to use for Spherical linear interpolation)
               vec1 = image_code1/np.linalg.norm(image_code1)
               vec2 = image_code2/np.linalg.norm(image_code2)
               vec2 = np.transpose(vec2)
               omega = np.arccos(np.clip(np.dot(vec1, vec2), -1, 1))
               so = np.sin(omega) 
                  
               # Combine the latent codes
               for p in pvals:
                      
                  # SIMPLE LATENT-SPACE LINEAR INTERPOLATION BETWEEN CLOSELY-RELATED CLASS PAIRS (SLI-CP)
                  new_code_slicp = np.multiply(p,image_code1) + np.multiply((1-p),image_code2)
                  new_label_slicp = np.multiply(p,label1) + np.multiply((1-p),label2)
                  new_label_slicp = new_label_slicp.reshape(1,1,NUM_CLASSES)

                  sample_slicp = np.zeros((1, N_CHANNELS, HEIGHT, WIDTH), dtype='int32')

                  # Generate SLI-CP sample
                  for y in xrange(HEIGHT):
                     for x in xrange(WIDTH):
                        for ch in xrange(N_CHANNELS):
                           next_sample_slicp = dec1_fn(new_code_slicp, sample_sli, ch, y, x) 
                           sample_slicp[:,ch,y,x] = next_sample_slicp
                      
                  # Add each mixed example and label to an array to be exported as a numpy array at the end
                  x_augmentation_set_slicp = np.concatenate((x_augmentation_set_slicp, sample_slicp), axis=0)
                  y_augmentation_set_slicp = np.concatenate((y_augmentation_set_slicp, new_label_slicp), axis=0)
                
                  # Save the SLI-CP-mixed example as an image. Comment out this line if desired.
                  color_grid_vis(sample_slicp,
                                 1,1,
                                 'interpolation_slicp_classes{}and{}_pval{}_num{}.png'.format(class1,
                                                                                              class2,
                                                                                              p,
                                                                                              imagenum))

                  # SPHERICAL LATENT-SPACE INTERPOLATION BETWEEN CLOSELY-RELATED CLASS PAIRS (SLERP-CP)
                  if so == 0:
                     new_code_slerpcp = (1.0-p) * image_code1 + p * image_code2
                  else:
                     new_code_slerpcp = np.sin((1.0-p)*omega) / so * image_code1 + np.sin(p*omega) / so * image_code2
                        
                  new_label_slerpcp = np.multiply(p,label1) + np.multiply((1-p),label2)
                  new_label_slerpcp = new_label_slerpcp.reshape(1,1,NUM_CLASSES)

                  sample_slerpcp = np.zeros((1, N_CHANNELS, HEIGHT, WIDTH),dtype='int32')

                  # Generate Slerp-CP sample
                  for y in xrange(HEIGHT):
                     for x in xrange(WIDTH):
                        for ch in xrange(N_CHANNELS):
                           next_sample_slerpcp = dec1_fn(new_code_slerpcp, sample_slerpcp, ch, y, x) 
                           sample_slerpcp[:,ch,y,x] = next_sample_slerpcp
                            
                  x_augmentation_set_slerpcp = np.concatenate((x_augmentation_set_slerpcp, sample_slerpcp), axis=0)
                  y_augmentation_set_slerpcp = np.concatenate((y_augmentation_set_slerpcp, new_label_slerpcp), axis=0)
   
                  # Save the Slerp-mixed example as an image. Comment out this line if desired.
                  color_grid_vis(sample_slerpcp,1,1,
                                 'interpolation_slerpcp_classes{}and{}_pval{}_num{}.png'.format(class1,
                                                                                                class2,
                                                                                                p,
                                                                                                imagenum))
            # Remove the placeholder rows in the image and label arrays
            x_augmentation_array_slicp = np.delete(x_augmentation_set_slicp, (0), axis=0)
            y_augmentation_array_slicp = np.delete(y_augmentation_set_slicp, (0), axis=0)
            x_augmentation_array_slerpcp = np.delete(x_augmentation_set_slerpcp, (0), axis=0)
            y_augmentation_array_slerpcp = np.delete(y_augmentation_set_slerpcp, (0), axis=0)
            
            # Convert the image pixels to uint8
            x_augmentation_array_slicp = x_augmentation_array_slicp.astype(np.uint8)
            x_augmentation_array_slerpcp = x_augmentation_array_slerpcp.astype(np.uint8)

            # Save arrays containing the augmentation sets as .npy files
            np.save(OUT_DIR + '/' + 'x_augmentation_array_slicp', x_augmentation_array_slicp)
            np.save(OUT_DIR + '/' + 'y_augmentation_array_slicp', y_augmentation_array_slicp)
            np.save(OUT_DIR + '/' + 'x_augmentation_array_slerpcp', x_augmentation_array_slerpcp)
            np.save(OUT_DIR + '/' + 'y_augmentation_array_slerpcp', y_augmentation_array_slerpcp)   
                      
    # Run 
    if MODE == 'one_level': # As before, other options have been pruned from this code, as we did not use multi-level PixelVAE's for our final analyses
        prints=[
            ('alpha', alpha), 
            ('reconst', reconst_cost), 
            ('kl1', kl_cost_1)
        ]

    decayed_lr = tf.train.exponential_decay(
        LR,
        total_iters,
        LR_DECAY_AFTER,
        LR_DECAY_FACTOR,
        staircase=True
    )

    lib.sampling_loop_cifar_filter_3.sampling_loop( 
        session=session,
        inputs=[total_iters, all_images],
        inject_iteration=True,
        bn_vars=(bn_is_training, bn_stats_iter),
        cost=full_cost,
        stop_after=TIMES['stop_after'],
        prints=prints,
        optimizer=tf.train.AdamOptimizer(decayed_lr),
        train_data=train_data,
        test_data=dev_data,
        callback=generate_and_save_samples,
        callback_every=TIMES['callback_every'],
        test_every=TIMES['test_every'],
        save_checkpoints=True
    )
